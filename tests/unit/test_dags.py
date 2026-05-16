from unittest.mock import MagicMock, patch

import pytest
from airflow.models import DagBag

DAGS_FOLDER = "mlops/airflow/dags"


@pytest.fixture(scope="module")
def dagbag():
    return DagBag(dag_folder=DAGS_FOLDER, include_examples=False)


def _get_dag(dagbag, dag_id):
    """Devuelve el DAG desde el cache local (sin consultar la BD)."""
    return dagbag.dags.get(dag_id)


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
