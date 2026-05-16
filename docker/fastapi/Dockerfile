FROM python:3.11-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.9.4 /uv /uvx /bin/

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project --group serving

COPY config.py ./
COPY serving ./serving
COPY feature_engineering ./feature_engineering

RUN groupadd --system app && useradd --system --gid app --create-home app && chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import socket; s=socket.create_connection(('localhost',8000),timeout=3); s.close()" || exit 1

CMD ["sh", "-c", "uvicorn serving.app.main:app --host 0.0.0.0 --port 8000 --workers ${FASTAPI_WORKERS:-1} --timeout-keep-alive 30"]
