FROM python:3.11-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.9.4 /uv /uvx /bin/

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project --group inference

COPY ingestion ./ingestion
COPY config.py ./

RUN groupadd --system app && useradd --system --gid app --create-home app && chown -R app:app /app

USER app

CMD ["python", "-m", "ingestion.inference_consumer.main"]
