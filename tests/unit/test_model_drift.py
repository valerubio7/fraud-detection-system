# ruff: noqa: E402
import sys
from unittest.mock import MagicMock

# run_model_drift_report usa pandas DataFrames y sklearn. Solo limpiamos si es stub.
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

from mlops.evidently.model_drift import _MIN_LABELED_ROWS, run_model_drift_report


class TestRunModelDriftReportInsufficientData:
    def test_returns_no_sufficient_data_when_empty(self):
        result = run_model_drift_report({}, pd.DataFrame(columns=["prediction_proba", "prediction", "target"]))
        assert result.has_sufficient_data is False
        assert result.drift_detected is False
        assert result.f1_degradation is None

    def test_returns_no_sufficient_data_when_below_minimum(self):
        n = _MIN_LABELED_ROWS - 1
        df = pd.DataFrame(
            {
                "prediction_proba": np.random.rand(n),
                "prediction": np.zeros(n, dtype=int),
                "target": np.zeros(n, dtype=int),
            }
        )
        result = run_model_drift_report({}, df)
        assert result.has_sufficient_data is False


class TestRunModelDriftReportSufficientData:
    def _make_labeled(self, n: int, predictions: list, targets: list) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "prediction_proba": np.random.rand(n),
                "prediction": predictions,
                "target": targets,
            }
        )

    def test_no_drift_when_f1_degradation_is_small(self):
        n = _MIN_LABELED_ROWS
        df = self._make_labeled(n, predictions=[1] * n, targets=[1] * n)
        reference = {"f1_score": 1.0, "precision": 1.0, "recall": 1.0}
        result = run_model_drift_report(reference, df)
        assert result.has_sufficient_data is True
        assert result.drift_detected is False

    def test_drift_detected_when_f1_degrades_more_than_5pct(self):
        n = _MIN_LABELED_ROWS
        # Perfect reference but model predicts everything as 0 (misses all fraud)
        df = self._make_labeled(n, predictions=[0] * n, targets=[1] * n)
        reference = {"f1_score": 1.0, "precision": 1.0, "recall": 1.0}
        result = run_model_drift_report(reference, df)
        assert result.has_sufficient_data is True
        assert result.drift_detected is True
        assert result.f1_degradation is not None
        assert result.f1_degradation < -0.05

    def test_metrics_are_populated(self):
        n = _MIN_LABELED_ROWS
        df = self._make_labeled(n, predictions=[1] * n, targets=[1] * n)
        reference = {"f1_score": 0.8, "precision": 0.9, "recall": 0.7}
        result = run_model_drift_report(reference, df)
        assert result.reference_f1 == pytest.approx(0.8)
        assert result.current_f1 is not None
        assert result.current_precision is not None
        assert result.current_recall is not None

    def test_missing_reference_metrics_returns_none_for_degradation(self):
        n = _MIN_LABELED_ROWS
        df = self._make_labeled(n, predictions=[1] * n, targets=[1] * n)
        result = run_model_drift_report({}, df)
        assert result.f1_degradation is None
        assert result.drift_detected is False


class TestToFloat:
    def test_returns_none_for_none_input(self):
        from mlops.evidently.model_drift import _to_float

        assert _to_float(None) is None

    def test_returns_float_for_valid_string(self):
        from mlops.evidently.model_drift import _to_float

        assert _to_float("3.14") == pytest.approx(3.14)

    def test_returns_none_for_invalid_string(self):
        from mlops.evidently.model_drift import _to_float

        assert _to_float("not_a_number") is None


class TestFetchLabeledPredictions:
    def _make_mock_pg(self, rows):
        from unittest.mock import MagicMock

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = rows
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg = MagicMock()
        mock_pg.connect.return_value = mock_conn
        return mock_pg

    def test_returns_empty_dataframe_when_no_rows(self):
        import sys
        from unittest.mock import patch

        from mlops.evidently.model_drift import fetch_labeled_predictions

        mock_pg = self._make_mock_pg([])
        with patch.dict(sys.modules, {"psycopg2": mock_pg}):
            df = fetch_labeled_predictions(deployment_id=1)
        assert len(df) == 0
        assert "prediction_proba" in df.columns

    def test_returns_dataframe_with_rows(self):
        import sys
        from unittest.mock import patch

        from mlops.evidently.model_drift import fetch_labeled_predictions

        rows = [(0.8, True, 1), (0.2, False, 0), (0.9, True, 1)]
        mock_pg = self._make_mock_pg(rows)
        with patch.dict(sys.modules, {"psycopg2": mock_pg}):
            df = fetch_labeled_predictions(deployment_id=1)
        assert len(df) == 3
        assert df["target"].dtype in (np.int64, np.int32, int)

    def test_conn_is_always_closed(self):
        import sys
        from unittest.mock import patch

        from mlops.evidently.model_drift import fetch_labeled_predictions

        mock_pg = self._make_mock_pg([])
        with patch.dict(sys.modules, {"psycopg2": mock_pg}):
            fetch_labeled_predictions(deployment_id=1)
        mock_pg.connect.return_value.close.assert_called_once()
