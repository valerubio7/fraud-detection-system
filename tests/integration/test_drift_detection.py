"""Integration tests for Evidently drift detection, DriftReportStore, and evaluate_drift_action."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from testcontainers.postgres import PostgresContainer

POSTGRESQL_MIGRATION = Path("database/postgresql/migrations/001_initial_schema.sql")
POSTGRES_IMAGE = "postgres:15"

COLUMNS = ["tx_count_1h", "amount_sum_1h", "seconds_since_last_tx"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    """Distribuciones radicalmente distintas para forzar drift detectable."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "tx_count_1h": rng.integers(20, 80, size=n).astype(float),
            "amount_sum_1h": rng.uniform(5_000.0, 20_000.0, size=n),
            "seconds_since_last_tx": rng.uniform(0.1, 2.0, size=n),
        }
    )


# ---------------------------------------------------------------------------
# 1. Evidently run_data_drift_report (no Docker)
# ---------------------------------------------------------------------------


class TestDataDriftDetection:
    def test_detects_drift_between_different_distributions(self):
        """run_data_drift_report detecta drift real entre distribuciones distintas."""
        from mlops.evidently.data_drift import run_data_drift_report

        result = run_data_drift_report(_make_reference_df(), _make_drifted_df(), COLUMNS)

        assert result.dataset_drift is True
        assert result.drift_share > 0.0
        assert len(result.drifted_features) > 0

    def test_no_drift_on_identical_data(self):
        """run_data_drift_report no detecta drift global cuando los datos son idénticos."""
        from mlops.evidently.data_drift import run_data_drift_report

        reference_df = _make_reference_df()
        result = run_data_drift_report(reference_df, reference_df.copy(), COLUMNS)

        assert result.dataset_drift is False
        assert result.drift_share < 0.5

    def test_feature_results_contains_all_columns(self):
        """feature_results incluye una entrada por cada columna analizada."""
        from mlops.evidently.data_drift import FeatureDriftResult, run_data_drift_report

        result = run_data_drift_report(_make_reference_df(), _make_drifted_df(), COLUMNS)

        for col in COLUMNS:
            assert col in result.feature_results
            fr = result.feature_results[col]
            assert isinstance(fr, FeatureDriftResult)
            assert isinstance(fr.drift_score, float)
            assert isinstance(fr.drift_detected, bool)
            assert isinstance(fr.stattest_name, str)


# ---------------------------------------------------------------------------
# 2. DriftReportStore → PostgreSQL (Docker)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDriftReportStoreIntegration:
    _pg_params: dict | None = None
    _deployment_id: int | None = None

    @pytest.fixture(autouse=True, scope="class")
    def setup_postgres(self):
        pg = PostgresContainer(
            image=POSTGRES_IMAGE,
            username="test_user",
            password="test_pass",
            dbname="test_pg",
        )
        with pg:
            pg_params = {
                "host": pg.get_container_host_ip(),
                "port": int(pg.get_exposed_port(5432)),
                "user": "test_user",
                "password": "test_pass",
                "dbname": "test_pg",
            }

            import psycopg2

            conn = psycopg2.connect(**pg_params)
            with conn.cursor() as cur:
                cur.execute(POSTGRESQL_MIGRATION.read_text())
            conn.commit()

            # FK: drift_reports references model_deployments
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO public.model_deployments
                        (model_name, version, mlflow_run_id,
                         training_data_from, training_data_to)
                    VALUES (%s, %s, %s, NOW() - INTERVAL '30 days', NOW())
                    RETURNING id
                    """,
                    ("DriftTestModel", "v1", "run_drift_abc"),
                )
                deployment_id = cur.fetchone()[0]
            conn.commit()
            conn.close()

            env_overrides = {
                "POSTGRES_HOST": pg_params["host"],
                "POSTGRES_PORT": str(pg_params["port"]),
                "POSTGRES_USER": pg_params["user"],
                "POSTGRES_PASSWORD": pg_params["password"],
                "POSTGRES_DB": pg_params["dbname"],
            }
            saved_env = {k: os.environ.get(k) for k in env_overrides}
            os.environ.update(env_overrides)

            TestDriftReportStoreIntegration._pg_params = pg_params
            TestDriftReportStoreIntegration._deployment_id = deployment_id

            try:
                yield
            finally:
                for k, old_v in saved_env.items():
                    if old_v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = old_v

    def test_save_persists_drift_report(self):
        """DriftReportStore.save() inserta una fila en drift_reports y devuelve su id."""
        from mlops.evidently.drift_store import DriftReportStore

        feature_drifts = {
            "tx_count_1h": {"drift_detected": True, "drift_score": 0.85},
            "amount_sum_1h": {"drift_detected": True, "drift_score": 0.92},
        }

        report_id = DriftReportStore().save(
            deployment_id=self.__class__._deployment_id,
            data_drift_score=0.67,
            feature_drifts=feature_drifts,
            model_drift_detected=False,
            alert_triggered=True,
            remediation_action="triggered_retraining_dag",
        )

        assert report_id is not None
        assert isinstance(report_id, int)

        import psycopg2

        conn = psycopg2.connect(**self.__class__._pg_params)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT drift_score, alert_triggered FROM public.drift_reports WHERE id = %s",
                (report_id,),
            )
            row = cur.fetchone()
        conn.close()

        assert row is not None
        assert abs(row[0] - 0.67) < 1e-6
        assert row[1] is True

    def test_save_returns_unique_ids(self):
        """Dos llamadas a save() generan ids distintos."""
        from mlops.evidently.drift_store import DriftReportStore

        feature_drifts = {"tx_count_1h": {"drift_detected": False, "drift_score": 0.05}}
        store = DriftReportStore()

        id1 = store.save(
            deployment_id=self.__class__._deployment_id,
            data_drift_score=0.10,
            feature_drifts=feature_drifts,
        )
        id2 = store.save(
            deployment_id=self.__class__._deployment_id,
            data_drift_score=0.12,
            feature_drifts=feature_drifts,
        )

        assert id1 != id2


# ---------------------------------------------------------------------------
# 3. evaluate_drift_action severity scaling (no Docker)
# ---------------------------------------------------------------------------


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
        """CRITICAL: feature crítica con drift > 0.20 Y model drift simultáneo."""
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
        """HIGH: feature crítica con drift > 0.20 pero sin model drift."""
        from mlops.evidently.drift_policy import evaluate_drift_action

        action = evaluate_drift_action(
            self._data(0.4, {"amount_sum_1h": 0.75}),
            self._model(drift_detected=False),
        )

        assert action.severity == "HIGH"
        assert action.alert_triggered is True
        assert action.trigger_retraining is True

    def test_high_when_only_model_drift(self):
        """HIGH: model drift detectado pero sin features críticas con score > 0.20."""
        from mlops.evidently.drift_policy import evaluate_drift_action

        action = evaluate_drift_action(
            self._data(0.1, {"non_critical": 0.9}),
            self._model(drift_detected=True, f1_degradation=0.08),
        )

        assert action.severity == "HIGH"
        assert action.alert_triggered is True
        assert action.trigger_retraining is True

    def test_warning_when_global_score_above_threshold(self):
        """WARNING: drift_score > 0.30 sin features críticas ni model drift."""
        from mlops.evidently.drift_policy import evaluate_drift_action

        action = evaluate_drift_action(
            self._data(0.45, {"non_critical": 0.05}),
            self._model(drift_detected=False),
        )

        assert action.severity == "WARNING"
        assert action.alert_triggered is True
        assert action.trigger_retraining is False
        assert action.remediation_action is None

    def test_info_when_within_acceptable_bounds(self):
        """INFO: drift_score bajo y features críticas por debajo del umbral."""
        from mlops.evidently.drift_policy import evaluate_drift_action

        action = evaluate_drift_action(
            self._data(0.05, {"tx_count_1h": 0.05}),
            self._model(drift_detected=False),
        )

        assert action.severity == "INFO"
        assert action.alert_triggered is False
        assert action.trigger_retraining is False
        assert action.remediation_action is None
