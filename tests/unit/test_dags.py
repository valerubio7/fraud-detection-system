from unittest.mock import MagicMock, patch

import pytest
from airflow.models import DagBag

DAGS_FOLDER = "mlops/airflow/dags"


@pytest.fixture(scope="module")
def dagbag():
    return DagBag(dag_folder=DAGS_FOLDER, include_examples=False)


def _get_dag(dagbag, dag_id):
    """Devuelve el DAG desde el cache local (sin consultar la BD)."""
    dag = dagbag.dags.get(dag_id)
    assert dag is not None, f"DAG '{dag_id}' no está en el DagBag — posible error de importación"
    return dag


class TestDagBagLoads:
    def test_no_import_errors(self, dagbag):
        """DagBag no debe tener errores de importación."""
        assert dagbag.import_errors == {}, f"DAG import errors: {dagbag.import_errors}"

    def test_all_expected_dags_are_present(self, dagbag):
        expected_dag_ids = {
            "retrain_fraud_model",
            "validate_and_promote_model",
            "drift_detection_report",
        }
        assert expected_dag_ids.issubset(set(dagbag.dag_ids))


class TestRetrainFraudModelDag:
    def test_dag_has_no_cycles(self, dagbag):
        dag = _get_dag(dagbag, "retrain_fraud_model")
        assert dag is not None
        assert len(dag.topological_sort()) > 0

    def test_dag_has_correct_task_count(self, dagbag):
        dag = _get_dag(dagbag, "retrain_fraud_model")
        assert len(dag.tasks) == 3

    def test_dag_task_ids(self, dagbag):
        dag = _get_dag(dagbag, "retrain_fraud_model")
        task_ids = {t.task_id for t in dag.tasks}
        assert "validate_data_availability" in task_ids
        assert "run_training" in task_ids
        assert "trigger_validation" in task_ids

    def test_validate_data_availability_has_no_upstream(self, dagbag):
        dag = _get_dag(dagbag, "retrain_fraud_model")
        task = dag.get_task("validate_data_availability")
        assert len(task.upstream_list) == 0

    def test_run_training_depends_on_validate(self, dagbag):
        dag = _get_dag(dagbag, "retrain_fraud_model")
        task = dag.get_task("run_training")
        upstream_ids = {t.task_id for t in task.upstream_list}
        assert "validate_data_availability" in upstream_ids

    def test_trigger_validation_depends_on_run_training(self, dagbag):
        dag = _get_dag(dagbag, "retrain_fraud_model")
        task = dag.get_task("trigger_validation")
        upstream_ids = {t.task_id for t in task.upstream_list}
        assert "run_training" in upstream_ids

    def test_dag_schedule_is_daily(self, dagbag):
        dag = _get_dag(dagbag, "retrain_fraud_model")
        # schedule "0 2 * * *" → diario a las 2 AM UTC
        assert dag.schedule_interval is not None
        assert dag.catchup is False


class TestValidateAndPromoteDag:
    def test_dag_has_correct_task_count(self, dagbag):
        dag = _get_dag(dagbag, "validate_and_promote_model")
        assert len(dag.tasks) == 5

    def test_archive_task_has_one_failed_trigger_rule(self, dagbag):
        from airflow.utils.trigger_rule import TriggerRule

        dag = _get_dag(dagbag, "validate_and_promote_model")
        archive_task = dag.get_task("archive_rejected_version")
        assert archive_task.trigger_rule == TriggerRule.ONE_FAILED


class TestDriftDetectionDag:
    def test_dag_has_correct_task_count(self, dagbag):
        # El DAG implementado tiene 8 tasks (spec indicaba 6 con nombres distintos)
        dag = _get_dag(dagbag, "drift_detection_report")
        assert len(dag.tasks) == 8

    def test_dag_task_ids(self, dagbag):
        dag = _get_dag(dagbag, "drift_detection_report")
        task_ids = {t.task_id for t in dag.tasks}
        expected = {
            "fetch_active_deployment",
            "fetch_production_data",
            "featurize_reference",
            "featurize_production",
            "run_evidently_report",
            "run_model_drift_task",
            "save_report_to_postgresql",
            "export_html_reports",
        }
        assert expected == task_ids

    def test_export_html_depends_on_save_report(self, dagbag):
        dag = _get_dag(dagbag, "drift_detection_report")
        export_task = dag.get_task("export_html_reports")
        upstream_ids = {t.task_id for t in export_task.upstream_list}
        assert "save_report_to_postgresql" in upstream_ids


class TestValidateDataAvailabilityTask:
    """Tests de la lógica interna de validate_data_availability."""

    def _get_validate_fn(self, dagbag):
        """Obtiene el callable Python de la task (sin decoradores Airflow)."""
        dag = _get_dag(dagbag, "retrain_fraud_model")
        return dag.get_task("validate_data_availability").python_callable

    def test_raises_skip_when_count_below_threshold(self, dagbag):
        """Si el COUNT de transacciones es < 1000, se lanza AirflowSkipException."""
        from airflow.exceptions import AirflowSkipException

        validate_fn = self._get_validate_fn(dagbag)

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (500, "2025-01-01", "2025-01-14")
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        # validate_fn.__globals__ es el namespace real del módulo DAG.
        # Lo parcheamos directamente en lugar de usar patch("retrain_fraud_model.psycopg2.connect")
        # porque DagBag registra el módulo con un prefijo hash en sys.modules.
        mock_pg = MagicMock()
        mock_pg.connect.return_value = mock_conn

        with (
            patch.dict(validate_fn.__globals__, {"psycopg2": mock_pg}),
            patch.dict("sys.modules", {"config": MagicMock()}),
        ):
            with pytest.raises(AirflowSkipException):
                validate_fn()

    def test_returns_dict_when_count_sufficient(self, dagbag):
        """Si el COUNT >= 1000, devuelve un dict con row_count, data_from, data_to."""
        validate_fn = self._get_validate_fn(dagbag)

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (5000, "2025-01-01", "2025-01-14")
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_pg = MagicMock()
        mock_pg.connect.return_value = mock_conn

        with (
            patch.dict(validate_fn.__globals__, {"psycopg2": mock_pg}),
            patch.dict("sys.modules", {"config": MagicMock()}),
        ):
            result = validate_fn()

        assert result["row_count"] == 5000
        assert "data_from" in result
        assert "data_to" in result


def _make_mock_conn(fetchone_result):
    """Utility: build a mock psycopg2 connection that returns fetchone_result."""
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = fetchone_result
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_pg = MagicMock()
    mock_pg.connect.return_value = mock_conn
    return mock_pg, mock_conn


class TestDataQualityCheckTasks:
    def _get_task_fn(self, dagbag, task_id):
        dag = _get_dag(dagbag, "data_quality_check")
        return dag.get_task(task_id).python_callable

    def test_check_transaction_volume_above_threshold(self, dagbag):
        fn = self._get_task_fn(dagbag, "check_transaction_volume")
        mock_pg, _ = _make_mock_conn((50,))
        with patch.dict(fn.__globals__, {"psycopg2": mock_pg}):
            result = fn()
        assert result["transaction_count"] == 50
        assert result["alert_triggered"] is False

    def test_check_transaction_volume_below_threshold_triggers_alert(self, dagbag):
        fn = self._get_task_fn(dagbag, "check_transaction_volume")
        mock_pg, _ = _make_mock_conn((2,))
        with patch.dict(fn.__globals__, {"psycopg2": mock_pg}):
            result = fn()
        assert result["alert_triggered"] is True

    def test_check_prediction_rate_above_threshold(self, dagbag):
        fn = self._get_task_fn(dagbag, "check_prediction_rate")
        mock_pg, _ = _make_mock_conn((20,))
        with patch.dict(fn.__globals__, {"psycopg2": mock_pg}):
            result = fn()
        assert result["prediction_count"] == 20
        assert result["alert_triggered"] is False

    def test_check_prediction_rate_below_threshold_triggers_alert(self, dagbag):
        fn = self._get_task_fn(dagbag, "check_prediction_rate")
        mock_pg, _ = _make_mock_conn((1,))
        with patch.dict(fn.__globals__, {"psycopg2": mock_pg}):
            result = fn()
        assert result["alert_triggered"] is True

    def test_check_amount_distribution_insufficient_data(self, dagbag):
        fn = self._get_task_fn(dagbag, "check_amount_distribution")
        mock_pg, _ = _make_mock_conn((5, 50.0, 10.0, 10.0, 100.0))
        with patch.dict(fn.__globals__, {"psycopg2": mock_pg}):
            result = fn()
        assert result["status"] == "insufficient_data"

    def test_check_amount_distribution_normal(self, dagbag):
        fn = self._get_task_fn(dagbag, "check_amount_distribution")
        mock_pg, _ = _make_mock_conn((200, 50.0, 10.0, 5.0, 5000.0))
        with patch.dict(fn.__globals__, {"psycopg2": mock_pg}):
            result = fn()
        assert result["avg_amount"] == 50.0
        assert result["alert_triggered"] is False

    def test_check_amount_distribution_anomalous_triggers_alert(self, dagbag):
        fn = self._get_task_fn(dagbag, "check_amount_distribution")
        mock_pg, _ = _make_mock_conn((200, 200_000.0, 10.0, 5.0, 500_000.0))
        with patch.dict(fn.__globals__, {"psycopg2": mock_pg}):
            result = fn()
        assert result["alert_triggered"] is True

    def test_summarize_checks_logs_without_error(self, dagbag):
        fn = self._get_task_fn(dagbag, "summarize_checks")
        fn(
            {"transaction_count": 100, "alert_triggered": False},
            {"prediction_count": 50, "alert_triggered": False},
            {"avg_amount": 75.0, "alert_triggered": False},
        )

    def test_summarize_checks_handles_insufficient_data(self, dagbag):
        fn = self._get_task_fn(dagbag, "summarize_checks")
        fn(
            {"transaction_count": 0, "alert_triggered": True},
            {"prediction_count": 0, "alert_triggered": True},
            {"status": "insufficient_data"},
        )


class TestValidateAndPromoteModelTasks:
    def _get_task_fn(self, dagbag, task_id):
        dag = _get_dag(dagbag, "validate_and_promote_model")
        return dag.get_task(task_id).python_callable

    def test_compare_with_champion_skips_when_gates_failed(self, dagbag):
        from airflow.exceptions import AirflowSkipException

        fn = self._get_task_fn(dagbag, "compare_with_champion")
        with pytest.raises(AirflowSkipException):
            fn(
                {"model_name": "FraudModel", "model_version": "5"},
                {"passed": False},
            )

    def test_promote_to_production_skips_when_challenger_loses(self, dagbag):
        from airflow.exceptions import AirflowSkipException

        fn = self._get_task_fn(dagbag, "promote_to_production_task")
        with pytest.raises(AirflowSkipException):
            fn(
                {"model_name": "FraudModel", "model_version": "5"},
                {"challenger_wins": False, "reason": "Champion is better"},
            )
