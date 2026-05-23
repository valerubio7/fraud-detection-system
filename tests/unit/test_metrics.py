# ruff: noqa: E402
import sys
from unittest.mock import MagicMock

# sklearn usa is_pandas_df() que requiere pd.DataFrame real. Solo limpiamos si es stub.
for _stub in ["pandas"]:
    _mod = sys.modules.get(_stub)
    if isinstance(_mod, MagicMock):
        sys.modules.pop(_stub, None)
        for _key in list(sys.modules):
            if _key.startswith(_stub + "."):
                sys.modules.pop(_key, None)

import numpy as np
import pytest

from model.utils.metrics import compute_threshold_metrics, evaluate_model, find_optimal_threshold


class TestComputeThresholdMetrics:
    def _make_data(self):
        y_true = np.array([0, 0, 1, 1, 1])
        proba = np.array([0.1, 0.4, 0.6, 0.8, 0.9])
        return y_true, proba

    def test_returns_expected_keys(self):
        y_true, proba = self._make_data()
        result = compute_threshold_metrics(y_true, proba, np.array([0.5]))
        assert set(result.keys()) == {"thresholds", "precision", "recall", "f1"}

    def test_one_entry_per_threshold(self):
        y_true, proba = self._make_data()
        thresholds = np.array([0.3, 0.5, 0.7])
        result = compute_threshold_metrics(y_true, proba, thresholds)
        assert len(result["thresholds"]) == 3
        assert len(result["precision"]) == 3
        assert len(result["recall"]) == 3
        assert len(result["f1"]) == 3

    def test_threshold_1_0_predicts_nothing_as_fraud(self):
        y_true, proba = self._make_data()
        result = compute_threshold_metrics(y_true, proba, np.array([1.0]))
        assert result["precision"][0] == pytest.approx(0.0)
        assert result["recall"][0] == pytest.approx(0.0)

    def test_threshold_0_0_predicts_everything_as_fraud(self):
        y_true, proba = self._make_data()
        result = compute_threshold_metrics(y_true, proba, np.array([0.0]))
        assert result["recall"][0] == pytest.approx(1.0)


class TestFindOptimalThreshold:
    def test_returns_threshold_and_metrics(self):
        y_true = np.array([0, 0, 1, 1])
        proba = np.array([0.1, 0.3, 0.7, 0.9])
        thresholds = np.linspace(0.1, 0.9, 5)
        threshold, metrics = find_optimal_threshold(y_true, proba, thresholds)
        assert isinstance(threshold, float)
        assert "thresholds" in metrics

    def test_prefers_lower_fn_cost_when_fn_is_expensive(self):
        y_true = np.array([0, 1, 1, 1])
        proba = np.array([0.4, 0.5, 0.6, 0.7])
        thresholds = np.array([0.3, 0.6])
        threshold, _ = find_optimal_threshold(
            y_true, proba, thresholds, cost_false_negative=1000.0, cost_false_positive=1.0
        )
        # With very expensive FN, the lower threshold (catches more fraud) should win
        assert threshold == pytest.approx(0.3)


class TestEvaluateModel:
    @pytest.fixture
    def mock_model(self):
        from unittest.mock import MagicMock

        model = MagicMock()
        # predict_proba returns [[prob_negative, prob_positive], ...]
        model.predict_proba.return_value = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]])
        return model

    def test_returns_all_expected_keys(self, mock_model):
        y_test = np.array([0, 0, 1, 1])
        result = evaluate_model(mock_model, None, y_test, threshold=0.5)
        expected_keys = {
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "pr_auc",
            "confusion_matrix",
            "fpr",
            "fnr",
            "threshold",
            "cost_false_negative",
            "cost_false_positive",
            "total_cost",
            "cost_per_transaction",
            "fraud_detected_pct",
            "legitimate_blocked_pct",
        }
        assert expected_keys == set(result.keys())

    def test_perfect_classifier_has_zero_cost(self, mock_model):
        y_test = np.array([0, 0, 1, 1])
        result = evaluate_model(mock_model, None, y_test, threshold=0.5)
        assert result["confusion_matrix"]["fn"] == 0
        assert result["confusion_matrix"]["fp"] == 0
        assert result["total_cost"] == pytest.approx(0.0)

    def test_threshold_affects_predictions(self, mock_model):
        y_test = np.array([0, 0, 1, 1])
        result_high = evaluate_model(mock_model, None, y_test, threshold=0.9)
        result_low = evaluate_model(mock_model, None, y_test, threshold=0.1)
        # At threshold=0.9 no fraud predicted → recall=0
        assert result_high["recall"] == pytest.approx(0.0)
        # At threshold=0.1 everything predicted fraud → recall=1
        assert result_low["recall"] == pytest.approx(1.0)

    def test_fraud_detected_pct_equals_recall(self, mock_model):
        y_test = np.array([0, 0, 1, 1])
        result = evaluate_model(mock_model, None, y_test, threshold=0.5)
        assert result["fraud_detected_pct"] == pytest.approx(result["recall"])
