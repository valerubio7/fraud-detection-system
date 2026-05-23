FROM apache/airflow:2.8.4-python3.11

ENV UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy

USER root

COPY --from=ghcr.io/astral-sh/uv:0.9.4 /uv /bin/

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --only-group airflow

USER airflow
