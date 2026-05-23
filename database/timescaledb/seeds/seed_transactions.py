#!/usr/bin/env python3
import argparse
import os
import random
from collections import Counter

import psycopg2
from psycopg2.extras import execute_values

from streaming.producer.generator import FraudPatternGenerator, LegitimateTransactionGenerator

FRAUD_PATTERNS = ["amount_anomaly", "unusual_country", "high_frequency", "unknown_merchant"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed synthetic transactions into TimescaleDB.")
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--fraud-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--num-users", type=int, default=200)
    parser.add_argument("--num-merchants", type=int, default=50)
    return parser.parse_args()


def get_db_config() -> dict:
    return {
        "host": os.getenv("TIMESCALE_HOST", "localhost"),
        "port": int(os.getenv("TIMESCALE_PORT", "5432")),
        "user": os.getenv("TIMESCALE_USER", "postgres"),
        "password": os.getenv("TIMESCALE_PASSWORD", "postgres"),
        "dbname": os.getenv("TIMESCALE_DB", "timescaledb"),
    }


def transaction_to_row(tx, is_fraud: bool) -> tuple:
    return (
        tx.transaction_id,
        tx.user_id,
        tx.merchant_id,
        tx.merchant_category,
        tx.amount,
        tx.country,
        tx.timestamp,
        tx.device_type,
        tx.ip_hash,
        is_fraud,
        None,
        None,
    )


def generate_fraud(
    fraud_gen: FraudPatternGenerator,
    legit_gen: LegitimateTransactionGenerator,
    pattern: str,
    rng: random.Random,
    target: int,
) -> list:
    if target <= 0:
        return []

    transactions = []
    generated = 0

    if pattern == "high_frequency":
        while generated < target:
            user = legit_gen.sample_user(rng)
            count = min(rng.randint(5, 8), max(5, target - generated + 5))
            txs = fraud_gen.apply_high_frequency(user, count=count)
            transactions.extend(txs)
            generated += len(txs)
        return transactions

    for _ in range(target):
        user = legit_gen.sample_user(rng)
        if pattern == "amount_anomaly":
            tx = fraud_gen.apply_amount_anomaly(user, [])
        elif pattern == "unusual_country":
            tx = fraud_gen.apply_unusual_country(user, set())
        elif pattern == "unknown_merchant":
            tx = fraud_gen.apply_unknown_merchant_high_amount(user, set())
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
        transactions.append(tx)

    return transactions


def insert_transactions(connection, rows: list[tuple], batch_size: int) -> None:
    insert_sql = (
        "INSERT INTO public.transactions "
        "(transaction_id, user_id, merchant_id, merchant_category, amount, country, "
        '"timestamp", device_type, ip_hash, is_fraud, model_score, latency_ms) '
        "VALUES %s "
        "ON CONFLICT (transaction_id, timestamp) DO NOTHING"
    )
    total = len(rows)
    inserted = 0
    try:
        with connection.cursor() as cursor:
            for start in range(0, total, batch_size):
                batch = rows[start : start + batch_size]
                execute_values(cursor, insert_sql, batch)
                connection.commit()
                inserted += len(batch)
                if inserted % 1000 == 0 or inserted == total:
                    print(f"Inserted {inserted}/{total} transactions")
    except psycopg2.Error:
        connection.rollback()
        raise


def print_summary(rows: list[tuple]) -> None:
    total = len(rows)
    fraud_total = sum(1 for r in rows if r[9])
    countries = Counter(r[5] for r in rows)
    timestamps = [r[6] for r in rows]
    min_ts = min(timestamps)
    max_ts = max(timestamps)
    print("\nSummary")
    print(f"Total inserted: {total}")
    print(f"Total fraud: {fraud_total}")
    print("Country distribution:")
    for country, cnt in countries.most_common():
        print(f"  {country}: {cnt} ({cnt / total * 100:.1f}%)")
    print(f"Date range: {min_ts.isoformat()} to {max_ts.isoformat()}")


def validate_args(args: argparse.Namespace) -> None:
    if args.count <= 0:
        raise SystemExit("--count must be greater than 0")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than 0")
    if args.fraud_rate < 0 or args.fraud_rate > 1:
        raise SystemExit("--fraud-rate must be between 0 and 1")


def main() -> None:
    args = parse_args()
    validate_args(args)

    legit_gen = LegitimateTransactionGenerator(
        seed=args.seed,
        user_count=args.num_users,
        merchant_count=args.num_merchants,
    )
    fraud_gen = FraudPatternGenerator(legit_gen, seed=args.seed + 1)
    rng = random.Random(args.seed + 2)

    fraud_total = min(max(int(round(args.count * args.fraud_rate)), 0), args.count)

    base = fraud_total // 4
    pattern_counts = [base, base, base, base]
    for i in range(fraud_total - base * 4):
        pattern_counts[i] += 1

    rows: list[tuple] = []

    fraud_txs: list = []
    for pattern, target in zip(FRAUD_PATTERNS, pattern_counts, strict=False):
        txs = generate_fraud(fraud_gen, legit_gen, pattern, rng, target)
        fraud_txs.extend(txs)
        rows.extend(transaction_to_row(tx, True) for tx in txs)

    fraud_gen.update_context(fraud_txs)
    legit_total = max(0, args.count - len(rows))
    legit_txs = legit_gen.generate_batch(legit_total)
    rows.extend(transaction_to_row(tx, False) for tx in legit_txs)

    with psycopg2.connect(**get_db_config()) as connection:
        insert_transactions(connection, rows, args.batch_size)

    print_summary(rows)


if __name__ == "__main__":
    main()
