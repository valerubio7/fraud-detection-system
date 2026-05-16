"""
Stubs de módulos pesados que no forman parte del grupo 'testing'.
Se inyectan en sys.modules antes de que pytest recolecte los tests,
por lo que los imports al nivel de módulo en serving/ no fallan.
"""

import sys
from unittest.mock import MagicMock

for _mod in [
    "joblib",
    "mlflow",
    "mlflow.tracking",
    "psycopg2",
    "asyncpg",
    "redis",
    "redis.exceptions",
    "prometheus_fastapi_instrumentator",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
