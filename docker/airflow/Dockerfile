FROM apache/airflow:2.8.4-python3.11

ENV UV_LINK_MODE=copy

USER root

COPY --from=ghcr.io/astral-sh/uv:0.9.4 /uv /bin/

COPY pyproject.toml uv.lock ./

RUN uv export --frozen --only-group airflow --no-hashes -o /tmp/reqs-airflow.txt

COPY offline_features /opt/airflow/project/offline_features
COPY model /opt/airflow/project/model
COPY mlops /opt/airflow/project/mlops

ENV PYTHONPATH=/opt/airflow/project

USER airflow

RUN pip install --no-cache-dir -r /tmp/reqs-airflow.txt
