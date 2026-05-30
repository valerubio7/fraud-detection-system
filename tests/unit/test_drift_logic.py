# ruff: noqa: E402
import sys
from unittest.mock import MagicMock

# Evidently requiere pandas real, no el stub.
for _stub in ["pandas"]:
    _mod = sys.modules.get(_stub)
    if isinstance(_mod, MagicMock):
        sys.modules.pop(_stub, None)
        for _key in list(sys.modules):
            if _key.startswith(_stub + "."):
                sys.modules.pop(_key, None)

import numpy as np
import pandas as pd

COLUMNS = ["tx_count_1h", "amount_sum_1h", "seconds_since_last_tx"]


def _make_reference_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "tx_count_1h": rng.integers(0, 5, size=n).astype(float),
            "amount_sum_1h": rng.uniform(10.0, 100.0, size=n),
            "seconds_since_last_tx": rng.uniform(60.0, 3600.0, size=n),
        }
    )


def _make_drifted_df(n: int = 500, seed: int = 99) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "tx_count_1h": rng.integers(20, 80, size=n).astype(float),
            "amount_sum_1h": rng.uniform(5_000.0, 20_000.0, size=n),
            "seconds_since_last_tx": rng.uniform(0.1, 2.0, size=n),
        }
    )


class TestDataDriftDetection:
    def test_detects_drift_between_different_distributions(self):
        from mlops.evidently.data_drift import run_data_drift_report

        result = run_data_drift_report(_make_reference_df(), _make_drifted_df(), COLUMNS)

        assert result.dataset_drift is True
        assert result.drift_share > 0.0
        assert len(result.drifted_features) > 0

    def test_no_drift_on_identical_data(self):
        from mlops.evidently.data_drift import run_data_drift_report

        reference_df = _make_reference_df()
        result = run_data_drift_report(reference_df, reference_df.copy(), COLUMNS)

        assert result.dataset_drift is False
        assert result.drift_share < 0.5

    def test_feature_results_contains_all_columns(self):
        from mlops.evidently.data_drift import FeatureDriftResult, run_data_drift_report

        result = run_data_drift_report(_make_reference_df(), _make_drifted_df(), COLUMNS)

        for col in COLUMNS:
            assert col in result.feature_results
            fr = result.feature_results[col]
            assert isinstance(fr, FeatureDriftResult)
            assert isinstance(fr.drift_score, float)
            assert isinstance(fr.drift_detected, bool)
            assert isinstance(fr.stattest_name, str)


class TestEvaluateDriftAction:
    @staticmethod
    def _data(drift_score: float, feature_scores: dict[str, float] | None = None) -> dict:
        return {
            "drift_score": drift_score,
            "feature_drifts": {name: {"drift_score": score} for name, score in (feature_scores or {}).items()},
        }

    @staticmethod
    def _model(drift_detected: bool = False, f1_degradation: float = 0.0) -> dict:
        return {"drift_detected": drift_detected, "f1_degradation": f1_degradation}

    def test_critical_when_critical_feature_and_model_drift(self):
        from mlops.evidently.drift_policy import evaluate_drift_action

        action = evaluate_drift_action(
            self._data(0.5, {"tx_count_1h": 0.9}),
            self._model(drift_detected=True, f1_degradation=0.12),
        )

        assert action.severity == "CRITICAL"
        assert action.alert_triggered is True
        assert action.trigger_retraining is True
        assert action.remediation_action == "triggered_retraining_dag"

    def test_high_when_only_critical_feature_drifted(self):
        from mlops.evidently.drift_policy import evaluate_drift_action

        action = evaluate_drift_action(
            self._data(0.4, {"amount_sum_1h": 0.75}),
            self._model(drift_detected=False),
        )

        assert action.severity == "HIGH"
        assert action.alert_triggered is True
        assert action.trigger_retraining is True

    def test_high_when_only_model_drift(self):
        from mlops.evidently.drift_policy import evaluate_drift_action

        action = evaluate_drift_action(
            self._data(0.1, {"non_critical": 0.9}),
            self._model(drift_detected=True, f1_degradation=0.08),
        )

        assert action.severity == "HIGH"
        assert action.alert_triggered is True
        assert action.trigger_retraining is True

    def test_warning_when_global_score_above_threshold(self):
        from mlops.evidently.drift_policy import evaluate_drift_action

        action = evaluate_drift_action(
            self._data(0.45, {"non_critical": 0.05}),
            self._model(drift_detected=False),
        )

        assert action.severity == "WARNING"
        assert action.alert_triggered is True
        assert action.trigger_retraining is True
        assert action.remediation_action == "triggered_retraining_dag"

    def test_info_when_within_acceptable_bounds(self):
        from mlops.evidently.drift_policy import evaluate_drift_action

        action = evaluate_drift_action(
            self._data(0.05, {"tx_count_1h": 0.05}),
            self._model(drift_detected=False),
        )

        assert action.severity == "INFO"
        assert action.alert_triggered is False
        assert action.trigger_retraining is False
        assert action.remediation_action is None
