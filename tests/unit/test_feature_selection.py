# ruff: noqa: E402
import sys
from unittest.mock import MagicMock, patch

for _stub in ["pandas"]:
    _mod = sys.modules.get(_stub)
    if isinstance(_mod, MagicMock):
        sys.modules.pop(_stub, None)
        for _key in list(sys.modules):
            if _key.startswith(_stub + "."):
                sys.modules.pop(_key, None)

import numpy as np
import pandas as pd
import pytest

from offline_features.feature_selection import (
    FeatureSelectionReport,
    compute_correlation_matrix,
    find_redundant_features,
    select_features,
)


def _make_df(correlated: bool = False, n: int = 50, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "feat_a": rng.standard_normal(n),
            "feat_b": rng.standard_normal(n),
            "feat_c": rng.standard_normal(n),
        }
    )
    if correlated:
        X["feat_d"] = X["feat_a"] * 0.99 + rng.standard_normal(n) * 0.01
    y = pd.Series(rng.integers(0, 2, n))
    return X, y


class TestComputeCorrelationMatrix:
    def test_self_correlation_is_one(self):
        X, _ = _make_df()
        corr = compute_correlation_matrix(X)
        for col in X.columns:
            assert corr.loc[col, col] == pytest.approx(1.0)

    def test_matrix_is_symmetric(self):
        X, _ = _make_df()
        corr = compute_correlation_matrix(X)
        assert corr.shape == (3, 3)
        for i in X.columns:
            for j in X.columns:
                assert corr.loc[i, j] == pytest.approx(corr.loc[j, i])

    def test_uncorrelated_features_have_low_correlation(self):
        X, _ = _make_df(correlated=False, n=200, seed=42)
        corr = compute_correlation_matrix(X)
        assert abs(corr.loc["feat_a", "feat_b"]) < 0.4


class TestFindRedundantFeatures:
    def test_no_redundant_pairs_when_independent(self):
        X, _ = _make_df(correlated=False, n=200, seed=42)
        pairs = find_redundant_features(X, correlation_threshold=0.85)
        assert pairs == []

    def test_detects_nearly_identical_features(self):
        X, _ = _make_df(correlated=True, n=100, seed=42)
        pairs = find_redundant_features(X, correlation_threshold=0.85)
        feature_names = {p[0] for p in pairs} | {p[1] for p in pairs}
        assert "feat_a" in feature_names or "feat_d" in feature_names

    def test_pairs_sorted_by_absolute_correlation(self):
        X, _ = _make_df(correlated=True, n=100, seed=42)
        pairs = find_redundant_features(X, correlation_threshold=0.0)
        for i in range(len(pairs) - 1):
            assert abs(pairs[i][2]) >= abs(pairs[i + 1][2])


class TestSelectFeatures:
    def _make_mock_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        features = X.columns.tolist()
        return pd.DataFrame({"feature": features, "importance": [0.1] * len(features)})

    def test_returns_feature_selection_report(self):
        X, y = _make_df(n=50, seed=42)
        mock_imp = self._make_mock_importance(X)
        with patch("offline_features.feature_selection.compute_xgboost_importance", return_value=mock_imp):
            report = select_features(X, y)
        assert isinstance(report, FeatureSelectionReport)

    def test_selected_features_is_subset_of_all(self):
        X, y = _make_df(n=50, seed=42)
        mock_imp = self._make_mock_importance(X)
        with patch("offline_features.feature_selection.compute_xgboost_importance", return_value=mock_imp):
            report = select_features(X, y)
        assert set(report.selected_features).issubset(set(report.all_features))

    def test_low_importance_feature_is_dropped(self):
        X, y = _make_df(n=50, seed=42)
        low_imp = pd.DataFrame({"feature": X.columns.tolist(), "importance": [0.001, 0.1, 0.1]})
        with patch("offline_features.feature_selection.compute_xgboost_importance", return_value=low_imp):
            report = select_features(X, y, importance_threshold=0.01)
        assert "feat_a" in report.dropped_features
        assert report.drop_reason["feat_a"] == "low_importance"

    def test_redundant_feature_is_dropped(self):
        X, y = _make_df(correlated=True, n=100, seed=42)
        mock_imp = self._make_mock_importance(X)
        with patch("offline_features.feature_selection.compute_xgboost_importance", return_value=mock_imp):
            report = select_features(X, y, correlation_threshold=0.85)
        all_dropped_reasons = set(report.drop_reason.values())
        if report.redundant_pairs:
            assert "redundant" in all_dropped_reasons
