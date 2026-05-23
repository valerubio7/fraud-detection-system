import logging
from uuid import UUID

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool

from streaming.models import Transaction

psycopg2.extras.register_uuid()

logger = logging.getLogger(__name__)


class TransactionStore:
    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        db: str,
        pool_min: int = 1,
        pool_max: int = 4,
        connect_timeout: int = 5,
    ) -> None:
        self._pool: ThreadedConnectionPool | None = None
        self._is_available = False

        try:
            self._pool = ThreadedConnectionPool(
                pool_min,
                pool_max,
                host=host,
                port=port,
                user=user,
                password=password,
                dbname=db,
                connect_timeout=connect_timeout,
            )
        except psycopg2.Error as exc:
            logger.warning("TimescaleDB unavailable at %s:%s: %s", host, port, exc)
        else:
            self._is_available = True
            logger.info("Connected to TimescaleDB at %s:%s", host, port)

    @property
    def is_available(self) -> bool:
        return self._is_available

    def write(self, transaction: Transaction) -> None:
        if not self._is_available or self._pool is None:
            logger.debug("TimescaleDB unavailable; skipping insert for %s", transaction.transaction_id)
            return

        try:
            transaction_id = UUID(transaction.transaction_id)
        except ValueError:
            logger.error("Invalid transaction_id UUID: %s", transaction.transaction_id)
            return

        conn = None
        try:
            conn = self._pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO public.transactions (
                        transaction_id,
                        user_id,
                        merchant_id,
                        merchant_category,
                        amount,
                        country,
                        device_type,
                        ip_hash,
                        timestamp,
                        is_fraud,
                        model_score,
                        latency_ms
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (transaction_id, timestamp) DO NOTHING
                    """,
                    (
                        transaction_id,
                        transaction.user_id,
                        transaction.merchant_id,
                        transaction.merchant_category,
                        transaction.amount,
                        transaction.country,
                        transaction.device_type,
                        transaction.ip_hash,
                        transaction.timestamp,
                        None,
                        None,
                        None,
                    ),
                )
            conn.commit()
        except psycopg2.Error as exc:
            logger.error("TimescaleDB insert failed for %s: %s", transaction.transaction_id, exc)
            if conn is not None:
                conn.rollback()
        finally:
            if conn is not None and self._pool is not None:
                self._pool.putconn(conn)

    def close(self) -> None:
        if self._pool is None:
            return
        try:
            self._pool.closeall()
        except psycopg2.Error as exc:
            logger.error("Failed to close TimescaleDB pool: %s", exc)


__all__ = ["TransactionStore"]
