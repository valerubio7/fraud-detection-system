FROM ghcr.io/mlflow/mlflow:v2.17.2

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:0.9.4 /uv /uvx /bin/

RUN uv pip install --no-cache-dir psycopg2-binary==2.9.9

WORKDIR /app
COPY mlops/mlflow/init_mlflow.py mlops/mlflow/init_mlflow.py
