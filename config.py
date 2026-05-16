from __future__ import annotations

from pathlib import Path
from typing import TypeVar
from urllib.parse import quote_plus

from pydantic import Field, ValidationError, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent / ".env"


class _BaseServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


class KafkaSettings(_BaseServiceSettings):
    broker_url: str = Field(default="kafka:29092", validation_alias="KAFKA_BROKER_URL")
    schema_registry_url: str = Field(
        default="http://schema-registry:8081",
        validation_alias="KAFKA_SCHEMA_REGISTRY_URL",
    )
    topics_raw: str = Field(default="transactions.raw", validation_alias="KAFKA_TOPICS_RAW")
    topics_features: str = Field(
        default="transactions.features",
        validation_alias="KAFKA_TOPICS_FEATURES",
    )
    topics_predictions: str = Field(
        default="transactions.predictions",
        validation_alias="KAFKA_TOPICS_PREDICTIONS",
    )
    topics_alerts: str = Field(
        default="transactions.fraud.alerts",
        validation_alias="KAFKA_TOPICS_ALERTS",
    )


class PostgreSQLSettings(_BaseServiceSettings):
    host: str = Field(default="postgresql", validation_alias="POSTGRES_HOST")
    port: int = Field(default=5432, validation_alias="POSTGRES_PORT")
    user: str = Field(default="fraud_metadata_user", validation_alias="POSTGRES_USER")
    password: str = Field(..., validation_alias="POSTGRES_PASSWORD")
    db: str = Field(default="fraud_metadata", validation_alias="POSTGRES_DB")

    @computed_field(return_type=str)
    @property
    def sqlalchemy_uri(self) -> str:
        user = quote_plus(self.user)
        password = quote_plus(self.password)
        return f"postgresql+psycopg2://{user}:{password}@{self.host}:{self.port}/{self.db}"


class TimescaleDBSettings(_BaseServiceSettings):
    host: str = Field(default="timescaledb", validation_alias="TIMESCALE_HOST")
    port: int = Field(default=5432, validation_alias="TIMESCALE_PORT")
    user: str = Field(default="fraud_timeseries_user", validation_alias="TIMESCALE_USER")
    password: str = Field(..., validation_alias="TIMESCALE_PASSWORD")
    db: str = Field(default="fraud_transactions_timeseries", validation_alias="TIMESCALE_DB")

    @computed_field(return_type=str)
    @property
    def sqlalchemy_uri(self) -> str:
        user = quote_plus(self.user)
        password = quote_plus(self.password)
        return f"postgresql+psycopg2://{user}:{password}@{self.host}:{self.port}/{self.db}"


class RedisSettings(_BaseServiceSettings):
    host: str = Field(default="redis", validation_alias="REDIS_HOST")
    port: int = Field(default=6379, validation_alias="REDIS_PORT")


class MLflowSettings(_BaseServiceSettings):
    tracking_uri: str = Field(default="http://mlflow:5000", validation_alias="MLFLOW_TRACKING_URI")
    backend_store_uri: str = Field(..., validation_alias="MLFLOW_BACKEND_STORE_URI")
    experiment_name: str = Field(
        default="fraud-detection-v1",
        validation_alias="MLFLOW_EXPERIMENT_NAME",
    )


class ModelSettings(_BaseServiceSettings):
    model_name: str = Field(default="FraudDetectionModel", validation_alias="MODEL_NAME")
    model_stage: str = Field(default="Production", validation_alias="MODEL_STAGE")
    fraud_score_threshold: float = Field(default=0.5, validation_alias="FRAUD_SCORE_THRESHOLD")
    slow_request_threshold_ms: float = Field(
        default=50.0, validation_alias="SLOW_REQUEST_THRESHOLD_MS"
    )


class AirflowSettings(_BaseServiceSettings):
    core_executor: str = Field(
        default="LocalExecutor",
        validation_alias="AIRFLOW__CORE__EXECUTOR",
    )
    database_sql_alchemy_conn: str = Field(
        ...,
        validation_alias="AIRFLOW__DATABASE__SQL_ALCHEMY_CONN",
    )
    core_fernet_key: str = Field(..., validation_alias="AIRFLOW__CORE__FERNET_KEY")
    webserver_secret_key: str = Field(
        ...,
        validation_alias="AIRFLOW__WEBSERVER__SECRET_KEY",
    )
    admin_user: str = Field(default="admin", validation_alias="AIRFLOW_ADMIN_USER")
    admin_password: str = Field(..., validation_alias="AIRFLOW_ADMIN_PASSWORD")


class GrafanaSettings(_BaseServiceSettings):
    security_admin_user: str = Field(default="admin", validation_alias="GF_SECURITY_ADMIN_USER")
    security_admin_password: str = Field(..., validation_alias="GF_SECURITY_ADMIN_PASSWORD")


SettingsT = TypeVar("SettingsT", bound=BaseSettings)


def _field_env_name(settings_cls: type[BaseSettings], field_name: str) -> str:
    field = settings_cls.model_fields.get(field_name)
    if field is None:
        return field_name

    if isinstance(field.validation_alias, str):
        return field.validation_alias

    if isinstance(field.alias, str):
        return field.alias

    return field_name


def _load_settings(settings_cls: type[SettingsT]) -> SettingsT:
    try:
        return settings_cls()
    except ValidationError as exc:
        missing_env_vars: list[str] = []
        for error in exc.errors():
            if error.get("type") != "missing":
                continue

            loc = error.get("loc", ())
            if not loc:
                continue

            missing_env_vars.append(_field_env_name(settings_cls, str(loc[0])))

        if missing_env_vars:
            missing_text = ", ".join(sorted(set(missing_env_vars)))
            raise RuntimeError(
                f"Error de configuracion en {settings_cls.__name__}. "
                f"Faltan variables obligatorias: {missing_text}"
            ) from exc

        raise RuntimeError(f"Error de configuracion en {settings_cls.__name__}: {exc}") from exc


_SETTINGS_REGISTRY: dict[str, type[BaseSettings]] = {
    "kafka_settings": KafkaSettings,
    "postgres_settings": PostgreSQLSettings,
    "timescaledb_settings": TimescaleDBSettings,
    "redis_settings": RedisSettings,
    "mlflow_settings": MLflowSettings,
    "model_settings": ModelSettings,
    "airflow_settings": AirflowSettings,
    "grafana_settings": GrafanaSettings,
}


def __getattr__(name: str) -> BaseSettings:
    if name in _SETTINGS_REGISTRY:
        value = _load_settings(_SETTINGS_REGISTRY[name])
        globals()[name] = value
        return value  # type: ignore[return-value]
    raise AttributeError(f"module 'config' has no attribute {name!r}")


__all__ = [
    "KafkaSettings",
    "PostgreSQLSettings",
    "TimescaleDBSettings",
    "RedisSettings",
    "MLflowSettings",
    "ModelSettings",
    "AirflowSettings",
    "GrafanaSettings",
    "kafka_settings",  # noqa: F822
    "postgres_settings",  # noqa: F822
    "timescaledb_settings",  # noqa: F822
    "redis_settings",  # noqa: F822
    "mlflow_settings",  # noqa: F822
    "model_settings",  # noqa: F822
    "airflow_settings",  # noqa: F822
    "grafana_settings",  # noqa: F822
]
