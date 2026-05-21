FROM apache/airflow:2.8.4-python3.11

ARG AIRFLOW_VERSION=2.8.4
ARG PYTHON_VERSION=3.11
ARG AIRFLOW_CONSTRAINTS_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300

USER root

COPY --from=ghcr.io/astral-sh/uv:0.9.4 /uv /bin/

RUN uv pip install --no-cache-dir \
    --constraint "${AIRFLOW_CONSTRAINTS_URL}" \
    "mlflow==2.17.2" \
    "evidently==0.4.23" \
    "psycopg2-binary>=2.9.0,<3.0.0" \
    "xgboost==2.0.3" \
    "scikit-learn>=1.4.0,<2.0.0" \
    "joblib>=1.3.0,<2.0.0" \
    "numpy>=1.24.0,<3.0.0" \
    "pandas>=2.1.0,<3.0.0"

USER airflow
