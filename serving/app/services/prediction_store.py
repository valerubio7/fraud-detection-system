import logging
import uuid

import asyncpg

_log = logging.getLogger(__name__)

_INSERT = """
INSERT INTO public.predictions_history
    (transaction_id, model_version_id, prediction_score, prediction_label, latency_ms)
VALUES ($1, $2, $3, $4, $5)
"""


class PredictionStore:
    def __init__(self, pool: asyncpg.Pool, deployment_id: int) -> None:
        self._pool = pool
        self._deployment_id = deployment_id

    async def save(
        self, transaction_id: str, prediction_score: float, prediction_label: bool, latency_ms: float
    ) -> None:
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    _INSERT,
                    uuid.UUID(transaction_id),
                    self._deployment_id,
                    prediction_score,
                    prediction_label,
                    latency_ms,
                )
        except Exception:
            _log.error("Error persisting prediction for transaction %s", transaction_id, exc_info=True)
