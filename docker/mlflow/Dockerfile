FROM ghcr.io/mlflow/mlflow:v2.17.2

ENV UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:0.9.4 /uv /uvx /bin/

RUN uv pip install --no-cache-dir psycopg2-binary==2.9.9

WORKDIR /app

COPY mlops/mlflow/init_mlflow.py mlops/mlflow/init_mlflow.py

CMD ["bash", "-c", "\
  mlflow server \
    --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
    --default-artifact-root mlflow-artifacts:/ \
    --artifacts-destination /mlflow/artifacts \
    --serve-artifacts \
    --host 0.0.0.0 \
    --port 5000 & \
  python mlops/mlflow/init_mlflow.py && \
  touch /tmp/mlflow_init_done && \
  wait"]
