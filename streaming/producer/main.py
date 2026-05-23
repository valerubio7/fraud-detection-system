import argparse
import itertools
import logging
import os
import random
import signal
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from streaming.models import Transaction
from streaming.producer.generator import FraudPatternGenerator, LegitimateTransactionGenerator, UserProfile
from streaming.producer.transaction_producer import TransactionProducer

logger = logging.getLogger(__name__)

DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "transaction_raw.avsc"
STATS_INTERVAL = 100
SCENARIO_FRAUD_RATE = 0.25
FRAUD_PATTERNS = ["amount_anomaly", "unusual_country", "high_frequency", "unknown_merchant"]


@dataclass
class ProducerStats:
    started_at: float
    total: int = 0
    legitimate: int = 0
    fraudulent: int = 0
    errors: int = 0


class StopEvent:
    def __init__(self) -> None:
        self._stop = False

    def set(self) -> None:
        self._stop = True

    def is_set(self) -> bool:
        return self._stop


class ProducerLoop:
    def __init__(
        self,
        producer: TransactionProducer,
        tps: int,
        duration_seconds: int,
        stats: ProducerStats,
        stop_event: StopEvent,
    ) -> None:
        self._producer = producer
        self._interval = 1.0 / tps
        self._stats = stats
        self._stop_event = stop_event
        self._deadline = stats.started_at + duration_seconds if duration_seconds > 0 else None

    def should_stop(self) -> bool:
        if self._stop_event.is_set():
            return True
        if self._deadline is None:
            return False
        return time.perf_counter() >= self._deadline

    def send(self, transaction: Transaction, is_fraud: bool) -> bool:
        if self.should_stop():
            return False

        start = time.perf_counter()
        try:
            self._producer.send(transaction)
        except Exception as exc:
            self._stats.errors += 1
            logger.error("Send failed for transaction %s: %s", transaction.transaction_id, exc)
        else:
            self._stats.total += 1
            if is_fraud:
                self._stats.fraudulent += 1
            else:
                self._stats.legitimate += 1

            if self._stats.total % STATS_INTERVAL == 0:
                log_stats(self._stats)

        elapsed = time.perf_counter() - start
        sleep_time = max(0.0, self._interval - elapsed)
        if sleep_time:
            time.sleep(sleep_time)

        return not self.should_stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fraud Detection Transaction Producer")

    parser.add_argument("--mode", choices=["live", "scenario"], default="live", help="Producer operation mode")
    parser.add_argument("--scenario", type=str, choices=FRAUD_PATTERNS, help="Fraud scenario to inject")
    parser.add_argument("--tps", type=int, default=10, help="Transactions per second (TPS)")
    parser.add_argument("--duration", type=int, default=0, help="Duration in seconds (0 = infinite)")
    parser.add_argument("--fraud-rate", type=float, default=0.02, help="Fraud rate (0.0 to 1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num-users", type=int, default=200, help="Number of users to simulate")
    parser.add_argument("--num-merchants", type=int, default=50, help="Number of available merchants")

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.tps <= 0:
        raise ValueError("--tps must be positive")
    if not 0.0 <= args.fraud_rate <= 1.0:
        raise ValueError("--fraud-rate must be between 0.0 and 1.0")
    if args.mode == "scenario" and not args.scenario:
        raise ValueError("--scenario is required when mode is scenario")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def log_stats(stats: ProducerStats) -> None:
    elapsed = time.perf_counter() - stats.started_at
    tps = stats.total / elapsed if elapsed > 0 else 0.0
    logger.info(
        "Sent=%s legit=%s fraud=%s errors=%s tps=%.2f",
        stats.total,
        stats.legitimate,
        stats.fraudulent,
        stats.errors,
        tps,
    )


def update_user_context(
    transaction: Transaction,
    history: dict[str, list[Transaction]],
    countries: dict[str, set[str]],
    merchants: dict[str, set[str]],
) -> None:
    history[transaction.user_id].append(transaction)
    countries[transaction.user_id].add(transaction.country)
    merchants[transaction.user_id].add(transaction.merchant_id)


def generate_fraud_transactions(
    fraud_generator: FraudPatternGenerator,
    pattern: str,
    user_profile: UserProfile,
    history: dict[str, list[Transaction]],
    countries: dict[str, set[str]],
    merchants: dict[str, set[str]],
    rng: random.Random,
) -> list[Transaction]:
    user_history = history.get(user_profile.user_id, [])
    user_countries = countries.get(user_profile.user_id, set())
    user_merchants = merchants.get(user_profile.user_id, set())

    if pattern == "amount_anomaly":
        return [fraud_generator.apply_amount_anomaly(user_profile, user_history)]
    if pattern == "unusual_country":
        return [fraud_generator.apply_unusual_country(user_profile, user_countries)]
    if pattern == "high_frequency":
        count = rng.randint(5, 8)
        return fraud_generator.apply_high_frequency(user_profile, count=count)
    if pattern == "unknown_merchant":
        return [fraud_generator.apply_unknown_merchant_high_amount(user_profile, user_merchants)]

    raise ValueError(f"Unsupported fraud pattern: {pattern}")


def run_generative_mode(
    loop: ProducerLoop,
    legit_generator: LegitimateTransactionGenerator,
    fraud_generator: FraudPatternGenerator,
    rng: random.Random,
    fraud_rate: float,
    scenario: str | None = None,
) -> ProducerStats:
    history: dict[str, list[Transaction]] = defaultdict(list)
    countries: dict[str, set[str]] = defaultdict(set)
    merchants: dict[str, set[str]] = defaultdict(set)
    pattern_cycle = itertools.cycle(FRAUD_PATTERNS)

    while not loop.should_stop():
        is_fraud = rng.random() < fraud_rate
        if is_fraud:
            pattern = next(pattern_cycle) if scenario is None or scenario == "mixed" else scenario
            user_profile = legit_generator.sample_user(rng)
            transactions = generate_fraud_transactions(
                fraud_generator,
                pattern,
                user_profile,
                history,
                countries,
                merchants,
                rng,
            )
            for transaction in transactions:
                update_user_context(transaction, history, countries, merchants)
                if not loop.send(transaction, is_fraud=True):
                    return loop._stats
        else:
            transaction = legit_generator.generate_transaction()
            fraud_generator.update_context([transaction])
            update_user_context(transaction, history, countries, merchants)
            if not loop.send(transaction, is_fraud=False):
                return loop._stats

    return loop._stats


def install_signal_handlers(stop_event: StopEvent) -> None:
    def _handle_signal(signum, _frame) -> None:
        logger.info("Received signal %s, shutting down", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)


def main() -> None:
    setup_logging()
    args = parse_args()
    validate_args(args)

    stop_event = StopEvent()
    install_signal_handlers(stop_event)

    producer = TransactionProducer(
        broker_url=os.getenv("KAFKA_BROKER_URL", "kafka:29092"),
        topic=os.getenv("KAFKA_TOPICS_RAW", "transactions.raw"),
        schema_path=str(DEFAULT_SCHEMA_PATH),
    )

    legit_generator = LegitimateTransactionGenerator(
        seed=args.seed,
        user_count=args.num_users,
        merchant_count=args.num_merchants,
    )
    fraud_generator = FraudPatternGenerator(legit_generator, seed=args.seed + 1)
    rng = random.Random(args.seed)

    stats = ProducerStats(started_at=time.perf_counter())
    loop = ProducerLoop(
        producer=producer, tps=args.tps, duration_seconds=args.duration, stats=stats, stop_event=stop_event
    )

    try:
        if args.mode == "scenario":
            run_generative_mode(
                loop=loop,
                legit_generator=legit_generator,
                fraud_generator=fraud_generator,
                rng=rng,
                fraud_rate=SCENARIO_FRAUD_RATE,
                scenario=args.scenario,
            )
        else:
            run_generative_mode(
                loop=loop,
                legit_generator=legit_generator,
                fraud_generator=fraud_generator,
                rng=rng,
                fraud_rate=args.fraud_rate,
            )
    finally:
        producer.close()
        log_stats(stats)


if __name__ == "__main__":
    main()
