FROM python:3.11-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}"

RUN groupadd --system app && useradd --system --gid app --create-home app

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.9.4 /uv /bin/

COPY --chown=app:app pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project --group serving

COPY --chown=app:app serving ./serving

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["sh", "-c", "exec uvicorn serving.app.main:app --host 0.0.0.0 --port 8000 --workers ${FASTAPI_WORKERS:-1} --timeout-keep-alive 30"]
