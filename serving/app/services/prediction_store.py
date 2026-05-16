from __future__ import annotations

import logging
import uuid

import psycopg2
import psycopg2.extras

import config

_log = logging.getLogger(__name__)

_INSERT = """
INSERT INTO public.predictions_history
    (transaction_id, model_version_id, prediction_score, prediction_label, latency_ms)
VALUES (%s, %s, %s, %s, %s)
"""


class PredictionStore:
    def __init__(self, deployment_id: int) -> None:
        self._deployment_id = deployment_id

    def save(
        self,
        transaction_id: str,
        prediction_score: float,
        prediction_label: bool,
        latency_ms: float,
    ) -> None:
        pg = config.postgres_settings
        conn = psycopg2.connect(
            host=pg.host,
            port=pg.port,
            user=pg.user,
            password=pg.password,
            dbname=pg.db,
        )
        try:
            psycopg2.extras.register_uuid(conn)
            with conn.cursor() as cur:
                cur.execute(
                    _INSERT,
                    (
                        uuid.UUID(transaction_id),
                        self._deployment_id,
                        prediction_score,
                        prediction_label,
                        latency_ms,
                    ),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            _log.error(
                "Error persisting prediction for transaction %s", transaction_id, exc_info=True
            )
        finally:
            conn.close()
