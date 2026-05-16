FROM apache/airflow:2.8.4-python3.11

ARG AIRFLOW_VERSION=2.8.4
ARG PYTHON_VERSION=3.11
ARG MLFLOW_VERSION=2.17.2
ARG EVIDENTLY_VERSION=0.4.23
ARG PSYCOPG2_VERSION=2.9.10
ARG AIRFLOW_CONSTRAINTS_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

USER root

COPY --from=ghcr.io/astral-sh/uv:0.9.4 /uv /uvx /bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN uv pip install --no-cache-dir \
    --constraint "${AIRFLOW_CONSTRAINTS_URL}" \
    "apache-airflow==${AIRFLOW_VERSION}" \
    "mlflow==${MLFLOW_VERSION}" \
    "evidently==${EVIDENTLY_VERSION}" \
    "psycopg2==${PSYCOPG2_VERSION}" \
    "xgboost==2.0.3" \
    "scikit-learn>=1.4.0,<2.0.0" \
    "joblib>=1.3.0,<2.0.0" \
    "numpy>=1.26.0,<3.0.0" \
    "pandas>=2.2.0,<3.0.0"

USER airflow
