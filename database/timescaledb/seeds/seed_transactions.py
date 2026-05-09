#!/usr/bin/env python3
import argparse
import os
import random
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta

import psycopg2
from psycopg2.extras import execute_values

from ingestion.producer.generator import (
    CATEGORY_MEDIANS,
    CATEGORY_SIGMA,
    DEVICE_TYPES,
    DEVICE_WEIGHTS,
    build_country_weights,
    build_hour_weights,
    build_merchants,
    choose_country,
    choose_device,
    choose_merchant,
    generate_amount,
    generate_ip_hash,
    generate_timestamp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed synthetic transactions into TimescaleDB.")
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--fraud-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=500)
    return parser.parse_args()


def get_db_config() -> dict:
    return {
        "host": os.getenv("TIMESCALE_HOST", "localhost"),
        "port": int(os.getenv("TIMESCALE_PORT", "5432")),
        "user": os.getenv("TIMESCALE_USER", "postgres"),
        "password": os.getenv("TIMESCALE_PASSWORD", "postgres"),
        "dbname": os.getenv("TIMESCALE_DB", "timescaledb"),
    }


def build_users(
    rng: random.Random,
    count: int,
    merchant_ids: list[str],
    countries: list[str],
    country_weights: list[float],
) -> list[dict]:
    users = []
    for idx in range(1, count + 1):
        users.append(
            {
                "user_id": f"user_{idx:04d}",
                "home_country": rng.choices(countries, weights=country_weights, k=1)[0],
                "preferred_device": rng.choices(DEVICE_TYPES, weights=DEVICE_WEIGHTS, k=1)[0],
                "spend_multiplier": max(0.4, min(2.5, rng.lognormvariate(0.0, 0.4))),
                "activity_weight": max(0.2, min(3.0, rng.lognormvariate(0.0, 0.6))),
                "frequent_merchants": rng.sample(merchant_ids, k=5),
            }
        )
    return users


def generate_transactions(
    rng: random.Random,
    count: int,
    users: list[dict],
    merchant_ids: list[str],
    merchant_categories: dict[str, str],
    countries: list[str],
    country_weights: list[float],
    hour_weights: list[float],
    transaction_namespace: uuid.UUID,
) -> list[dict]:
    now = datetime.now(UTC)
    user_weights = [u["activity_weight"] for u in users]
    transactions = []
    for idx in range(count):
        user = rng.choices(users, weights=user_weights, k=1)[0]
        merchant_id = choose_merchant(rng, user["frequent_merchants"], merchant_ids)
        category = merchant_categories[merchant_id]
        transactions.append(
            {
                "transaction_id": uuid.uuid5(transaction_namespace, f"transaction-{idx}"),
                "user_id": user["user_id"],
                "merchant_id": merchant_id,
                "merchant_category": category,
                "amount": generate_amount(
                    rng, category, user["spend_multiplier"], CATEGORY_MEDIANS, CATEGORY_SIGMA
                ),
                "country": choose_country(rng, user["home_country"], countries, country_weights),
                "timestamp": generate_timestamp(rng, now, hour_weights, days_back=30),
                "device_type": choose_device(rng, user["preferred_device"]),
                "ip_hash": generate_ip_hash(rng),
                "is_fraud": False,
                "model_score": None,
                "latency_ms": None,
            }
        )
    return transactions


def compute_user_average_amount(transactions: list[dict]) -> dict[str, float]:
    totals: defaultdict[str, float] = defaultdict(float)
    counts: defaultdict[str, int] = defaultdict(int)
    for tx in transactions:
        totals[tx["user_id"]] += float(tx["amount"])
        counts[tx["user_id"]] += 1
    return {uid: totals[uid] / counts[uid] for uid in totals}


def compute_user_seen_merchants(transactions: list[dict]) -> dict[str, set[str]]:
    seen: defaultdict[str, set] = defaultdict(set)
    for tx in transactions:
        seen[tx["user_id"]].add(tx["merchant_id"])
    return dict(seen)


def compute_user_home_countries(users: list[dict]) -> dict[str, str]:
    return {u["user_id"]: u["home_country"] for u in users}


def compute_user_indices(transactions: list[dict]) -> dict[str, list[int]]:
    indices: defaultdict[str, list] = defaultdict(list)
    for idx, tx in enumerate(transactions):
        indices[tx["user_id"]].append(idx)
    return dict(indices)


def apply_amount_anomaly(
    rng: random.Random,
    transactions: list[dict],
    available_indices: set[int],
    user_avg_amount: dict[str, float],
    count: int,
) -> int:
    if count <= 0 or not available_indices:
        return 0
    chosen = rng.sample(list(available_indices), k=min(count, len(available_indices)))
    for idx in chosen:
        tx = transactions[idx]
        avg = user_avg_amount.get(tx["user_id"], float(tx["amount"]))
        tx["amount"] = round(max(avg * rng.uniform(5.0, 10.0), float(tx["amount"])), 2)
        tx["is_fraud"] = True
        available_indices.remove(idx)
    return len(chosen)


def apply_unusual_country(
    rng: random.Random,
    transactions: list[dict],
    available_indices: set[int],
    user_home_countries: dict[str, str],
    countries: list[str],
    count: int,
) -> int:
    if count <= 0 or not available_indices:
        return 0
    applied = 0
    attempts = 0
    max_attempts = count * 20
    while applied < count and attempts < max_attempts and available_indices:
        idx = rng.choice(list(available_indices))
        tx = transactions[idx]
        home = user_home_countries.get(tx["user_id"], tx["country"])
        different = [c for c in countries if c != home]
        if not different:
            attempts += 1
            continue
        tx["country"] = rng.choice(different)
        tx["is_fraud"] = True
        available_indices.remove(idx)
        applied += 1
    return applied


def apply_high_frequency(
    rng: random.Random,
    transactions: list[dict],
    available_indices: set[int],
    user_indices: dict[str, list[int]],
    now: datetime,
    count: int,
) -> int:
    if count <= 0 or not available_indices:
        return 0
    remaining = count
    users = list(user_indices.keys())
    while remaining > 0:
        candidates = [
            (uid, [i for i in user_indices[uid] if i in available_indices]) for uid in users
        ]
        candidates = [(uid, idxs) for uid, idxs in candidates if len(idxs) >= 5]
        if not candidates:
            break
        uid, user_idxs = rng.choice(candidates)
        burst_size = min(remaining, rng.randint(5, 8), len(user_idxs))
        if burst_size < 5:
            break
        chosen = rng.sample(user_idxs, k=burst_size)
        base_time = transactions[chosen[0]]["timestamp"]
        for idx in chosen:
            tx = transactions[idx]
            new_time = base_time + timedelta(minutes=rng.randint(0, 29), seconds=rng.randint(0, 59))
            if new_time > now:
                new_time = now - timedelta(minutes=rng.randint(0, 29))
            tx["timestamp"] = new_time
            tx["is_fraud"] = True
            available_indices.remove(idx)
        remaining -= burst_size
    return count - remaining


def apply_unknown_merchant_high_amount(
    rng: random.Random,
    transactions: list[dict],
    available_indices: set[int],
    user_indices: dict[str, list[int]],
    user_seen_merchants: dict[str, set[str]],
    merchant_ids: list[str],
    merchant_categories: dict[str, str],
    count: int,
) -> int:
    if count <= 0 or not available_indices:
        return 0
    remaining = count
    users = list(user_indices.keys())
    all_merchants = set(merchant_ids)
    attempts = 0
    max_attempts = count * 50
    while remaining > 0 and attempts < max_attempts:
        uid = rng.choice(users)
        user_idxs = [i for i in user_indices[uid] if i in available_indices]
        if not user_idxs:
            attempts += 1
            continue
        unseen = list(all_merchants - user_seen_merchants[uid])
        if not unseen:
            attempts += 1
            continue
        idx = rng.choice(user_idxs)
        merchant_id = rng.choice(unseen)
        tx = transactions[idx]
        tx["merchant_id"] = merchant_id
        tx["merchant_category"] = merchant_categories[merchant_id]
        tx["amount"] = round(rng.uniform(300.0, 1200.0), 2)
        tx["is_fraud"] = True
        available_indices.remove(idx)
        user_seen_merchants[uid].add(merchant_id)
        remaining -= 1
    return count - remaining


def insert_transactions(connection, transactions: list[dict], batch_size: int) -> None:
    insert_sql = (
        "INSERT INTO public.transactions "
        "(transaction_id, user_id, merchant_id, merchant_category, amount, country, "
        '"timestamp", device_type, ip_hash, is_fraud, model_score, latency_ms) '
        "VALUES %s "
        "ON CONFLICT (transaction_id, timestamp) DO NOTHING"
    )
    total = len(transactions)
    inserted = 0
    try:
        with connection.cursor() as cursor:
            for start in range(0, total, batch_size):
                batch = transactions[start : start + batch_size]
                values = [
                    (
                        str(tx["transaction_id"]),
                        tx["user_id"],
                        tx["merchant_id"],
                        tx["merchant_category"],
                        tx["amount"],
                        tx["country"],
                        tx["timestamp"],
                        tx["device_type"],
                        tx["ip_hash"],
                        tx["is_fraud"],
                        tx["model_score"],
                        tx["latency_ms"],
                    )
                    for tx in batch
                ]
                execute_values(cursor, insert_sql, values)
                connection.commit()
                inserted += len(batch)
                if inserted % 1000 == 0 or inserted == total:
                    print(f"Inserted {inserted}/{total} transactions")
    except psycopg2.Error:
        connection.rollback()
        raise


def print_summary(transactions: list[dict]) -> None:
    total = len(transactions)
    fraud_total = sum(1 for tx in transactions if tx["is_fraud"])
    countries = Counter(tx["country"] for tx in transactions)
    min_ts = min(tx["timestamp"] for tx in transactions)
    max_ts = max(tx["timestamp"] for tx in transactions)
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

    rng = random.Random(args.seed)
    countries, country_weights = build_country_weights()
    hour_weights = build_hour_weights()
    merchant_ids, merchant_categories = build_merchants(rng, 50)
    users = build_users(rng, 200, merchant_ids, countries, country_weights)
    transaction_namespace = uuid.uuid5(uuid.NAMESPACE_DNS, f"seed-{args.seed}")

    transactions = generate_transactions(
        rng,
        args.count,
        users,
        merchant_ids,
        merchant_categories,
        countries,
        country_weights,
        hour_weights,
        transaction_namespace,
    )

    user_avg_amount = compute_user_average_amount(transactions)
    user_seen_merchants = compute_user_seen_merchants(transactions)
    user_home_countries = compute_user_home_countries(users)
    user_indices = compute_user_indices(transactions)

    fraud_total = min(max(int(round(args.count * args.fraud_rate)), 0), args.count)
    base = fraud_total // 4
    pattern_counts = [base, base, base, base]
    for i in range(fraud_total - base * 4):
        pattern_counts[i] += 1

    available_indices = set(range(len(transactions)))
    now = datetime.now(UTC)

    applied = [
        apply_amount_anomaly(
            rng, transactions, available_indices, user_avg_amount, pattern_counts[0]
        ),
        apply_unusual_country(
            rng, transactions, available_indices, user_home_countries, countries, pattern_counts[1]
        ),
        apply_high_frequency(
            rng, transactions, available_indices, user_indices, now, pattern_counts[2]
        ),
        apply_unknown_merchant_high_amount(
            rng,
            transactions,
            available_indices,
            user_indices,
            user_seen_merchants,
            merchant_ids,
            merchant_categories,
            pattern_counts[3],
        ),
    ]

    remaining = fraud_total - sum(applied)
    if remaining > 0:
        apply_amount_anomaly(rng, transactions, available_indices, user_avg_amount, remaining)

    with psycopg2.connect(**get_db_config()) as connection:
        insert_transactions(connection, transactions, args.batch_size)

    print_summary(transactions)


if __name__ == "__main__":
    main()
