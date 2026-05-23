"""
Stubs de módulos pesados que no forman parte del grupo 'testing'.
Se inyectan en sys.modules antes de que pytest recolecte los tests,
por lo que los imports al nivel de módulo en serving/ y mlops/ no fallan.
"""

import os
import sys
from unittest.mock import MagicMock

# --- Stubs de módulos sin instalar ---
for _mod in [
    "joblib",
    "mlflow",
    "mlflow.tracking",
    "psycopg2",
    "asyncpg",
    "redis",
    "redis.exceptions",
    "prometheus_fastapi_instrumentator",
    # pandas: importado a nivel de módulo en mlops/evidently/reference.py
    "pandas",
    # imblearn / xgboost: no instalados en el grupo 'testing'
    "imblearn",
    "imblearn.over_sampling",
    "xgboost",
    # confluent_kafka / fastavro: necesarios para streaming pero no en testing
    "confluent_kafka",
    "fastavro",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# --- Paths para DAGs y plugins de Airflow ---
# fraud_operators está en plugins/ (no en dags/ ni en site-packages)
for _path in ["mlops/airflow/plugins", "mlops/airflow/dags"]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

# --- Configuración mínima de Airflow para tests unitarios sin BD real ---
os.environ.setdefault("AIRFLOW_HOME", "/tmp/airflow_test_fraud")
os.environ.setdefault("AIRFLOW__DATABASE__SQL_ALCHEMY_CONN", "sqlite://")
