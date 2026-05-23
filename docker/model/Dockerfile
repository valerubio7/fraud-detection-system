FROM python:3.11-slim

ENV UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.9.4 /uv /bin/

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project --group model

COPY model ./model
COPY offline_features ./offline_features

RUN groupadd --system app && useradd --system --gid app --create-home app && chown -R app:app /app

USER app

CMD ["python", "model/pipeline/train.py"]
