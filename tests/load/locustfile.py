"""
Locust load test for the FastAPI /predict endpoint.

Run headless (CI):
    locust -f tests/load/locustfile.py --headless \
           -u 500 -r 50 --run-time 120s \
           --host http://localhost:8000 \
           --html tests/load/report.html

Run interactivo (UI en http://localhost:8089):
    locust -f tests/load/locustfile.py --host http://localhost:8000

Prerequisites:
    - Stack completo corriendo: docker compose up -d
    - Modelo en Production en MLflow
    - uv run --group testing locust -f tests/load/locustfile.py ...
"""

from locust import HttpUser, between, events, task

# Payload base: se reusan valores fijos para maximizar TPS
# (la API no requiere unicidad de transaction_id para el test de carga)
_BASE_FEATURES = {
    "tx_count_1h": 3.0,
    "tx_count_24h": 10.0,
    "tx_count_7d": 50.0,
    "amount_sum_1h": 300.0,
    "amount_sum_24h": 1000.0,
    "seconds_since_last_tx": 600.0,
    "amount_ratio_vs_user_avg": 1.2,
    "is_country_new": 0.0,
    "distinct_countries_seen": 3.0,
    "is_merchant_new": 0.0,
    "distinct_merchants_seen": 7.0,
}

_BASE_PAYLOAD = {
    "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
    "user_id": "load_test_user",
    "merchant_id": "merchant_load_test",
    "merchant_category": "grocery",
    "amount": 150.0,
    "country": "AR",
    "timestamp": "2025-01-15T14:30:00Z",
    "device_type": "mobile",
    "ip_hash": "load_test_hash",
    "features": _BASE_FEATURES,
}

_BATCH_PAYLOAD = {"items": [_BASE_PAYLOAD] * 10}


class FastAPIUser(HttpUser):
    """Simula un cliente enviando transacciones individuales a /predict."""

    wait_time = between(0.001, 0.05)  # ~20-1000 req/s por usuario

    @task(weight=9)
    def predict_single(self):
        """POST /predict — carga principal (90% del tráfico)."""
        with self.client.post(
            "/predict",
            json=_BASE_PAYLOAD,
            catch_response=True,
            name="/predict",
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:100]}")
                return
            process_time = response.headers.get("X-Process-Time-Ms")
            if process_time and float(process_time) > 200:
                # Warning individual — el P99 agregado se evalúa al final
                response.failure(f"Slow response: {process_time}ms > 200ms hard limit")

    @task(weight=1)
    def predict_batch(self):
        """POST /predict/batch — carga secundaria (10% del tráfico)."""
        self.client.post("/predict/batch", json=_BATCH_PAYLOAD, name="/predict/batch")

    def on_start(self):
        """Verificar que la API está healthy antes de iniciar carga."""
        response = self.client.get("/health")
        if response.status_code != 200:
            raise Exception("FastAPI no está healthy — abortando test")


@events.quitting.add_listener
def validate_thresholds(environment, **kwargs):
    """
    Evalúa los SLA al finalizar el test:
    - Error rate < 1% (idealmente 0%)
    - P99 de /predict < 100ms
    Si algún threshold falla, el proceso sale con código 1 (falla CI).
    """
    stats = environment.stats

    # Error rate
    total_fail_ratio = stats.total.fail_ratio
    if total_fail_ratio > 0.01:
        print(f"\n❌ FAIL: Error rate {total_fail_ratio:.2%} > 1% threshold")
        environment.process_exit_code = 1
    else:
        print(f"\n✅ Error rate: {total_fail_ratio:.2%}")

    # P99 de /predict
    predict_stats = stats.get("/predict", "POST")
    if predict_stats.num_requests == 0:
        print("⚠️  Sin requests a /predict — no se puede evaluar P99")
        environment.process_exit_code = 1
        return

    p99_ms = predict_stats.get_response_time_percentile(0.99)
    if p99_ms > 100:
        print(f"❌ FAIL: P99 latency {p99_ms:.1f}ms > 100ms SLA")
        environment.process_exit_code = 1
    else:
        print(f"✅ P99 latency: {p99_ms:.1f}ms")

    # Throughput mínimo (al menos 100 req/s con 500 usuarios)
    print(f"ℹ️  Throughput /predict: {stats.total.current_rps:.1f} req/s total")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Imprime un resumen legible al terminar."""
    stats = environment.stats
    print("\n" + "=" * 60)
    print("LOAD TEST SUMMARY")
    print("=" * 60)
    for entry in stats.entries.values():
        print(
            f"{entry.method:6} {entry.name:30} "
            f"reqs={entry.num_requests:6d} "
            f"fails={entry.num_failures:4d} "
            f"p50={entry.get_response_time_percentile(0.50):5.0f}ms "
            f"p95={entry.get_response_time_percentile(0.95):5.0f}ms "
            f"p99={entry.get_response_time_percentile(0.99):5.0f}ms"
        )
    print("=" * 60)
