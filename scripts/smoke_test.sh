#!/usr/bin/env bash
set -euo pipefail

# smoke_test.sh — Verificación post-deploy del stack completo.
#
# Uso:
#   ./scripts/smoke_test.sh           # verifica todo el stack
#
# Retorna exit code 0 si todos los checks pasan, 1 si alguno falla.
# Diseñado para ejecutarse automáticamente desde deploy.sh y en CI.

cd "$(dirname "$0")/.."

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

FAILURES=0

smoke_check() {
  local label="$1"
  local cmd="$2"
  printf "  %-45s" "${label}..."
  if eval "${cmd}" >/dev/null 2>&1; then
    printf "%s OK%s\n" "${GREEN}" "${RESET}"
  else
    printf "%s FAIL%s\n" "${RED}" "${RESET}"
    FAILURES=$((FAILURES + 1))
  fi
}

# Cargar variables de entorno
set -a
# shellcheck disable=SC1091
[[ -f .env ]] && source .env
set +a

AIRFLOW_ADMIN_USER="${AIRFLOW_ADMIN_USER:-admin}"
AIRFLOW_ADMIN_PASSWORD="${AIRFLOW_ADMIN_PASSWORD:-}"

print_step "Smoke test — Verificando servicios HTTP"

smoke_check "FastAPI /health (200 + status:ok)" \
  "curl -fsS http://localhost:8000/health | grep -q '\"status\":\"ok\"'"

smoke_check "FastAPI /model/info (modelo cargado)" \
  "curl -fsS http://localhost:8000/model/info | grep -q 'model_version'"

smoke_check "MLflow tracking server" \
  "curl -fsS http://localhost:5000/health"

smoke_check "Airflow webserver /health" \
  "curl -fsS http://localhost:8081/health | grep -q 'healthy'"

smoke_check "Grafana API /api/health" \
  "curl -fsS http://localhost:3000/api/health | grep -q '\"database\":\"ok\"'"

smoke_check "Prometheus /-/healthy" \
  "curl -fsS http://localhost:9090/-/healthy"

smoke_check "Kafka UI" \
  "curl -fsS http://localhost:8080"

print_step "Smoke test — Verificando topics Kafka"

for topic in transactions.raw transactions.features transactions.predictions transactions.fraud.alerts; do
  smoke_check "Topic: ${topic}" \
    "docker compose exec -T kafka kafka-topics --bootstrap-server localhost:29092 --list | grep -q '^${topic}$'"
done

print_step "Smoke test — Verificando DAGs de Airflow"

AIRFLOW_AUTH="${AIRFLOW_ADMIN_USER}:${AIRFLOW_ADMIN_PASSWORD}"
for dag_id in retrain_fraud_model validate_and_promote_model drift_detection_report data_quality_check; do
  smoke_check "DAG: ${dag_id}" \
    "curl -fsS -u '${AIRFLOW_AUTH}' http://localhost:8081/api/v1/dags/${dag_id} | grep -q '\"dag_id\"'"
done

print_step "Smoke test — Transacción de prueba POST /predict"

PREDICT_PAYLOAD='{
  "transaction_id": "smoke-test-tx-001",
  "user_id": "smoke_user",
  "merchant_id": "smoke_merchant",
  "merchant_category": "grocery",
  "amount": 99.99,
  "country": "AR",
  "timestamp": "2025-01-15T14:30:00Z",
  "device_type": "mobile",
  "ip_hash": "smoketest123",
  "features": {
    "tx_count_1h": 2.0,
    "tx_count_24h": 5.0,
    "tx_count_7d": 20.0,
    "amount_sum_1h": 150.0,
    "amount_sum_24h": 500.0,
    "seconds_since_last_tx": 600.0,
    "amount_ratio_vs_user_avg": 1.2,
    "is_country_new": 0.0,
    "distinct_countries_seen": 2.0,
    "is_merchant_new": 0.0,
    "distinct_merchants_seen": 5.0
  }
}'

printf "  %-45s" "POST /predict → PredictionResponse..."
RESPONSE=$(curl -fsS -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "${PREDICT_PAYLOAD}" 2>/dev/null) || { printf "%s FAIL%s (sin respuesta)\n" "${RED}" "${RESET}"; FAILURES=$((FAILURES + 1)); }

if [[ -n "${RESPONSE:-}" ]]; then
  SCORE=$(printf '%s' "${RESPONSE}" | grep -o '"prediction_score":[0-9.]*' | cut -d: -f2)
  LABEL=$(printf '%s' "${RESPONSE}" | grep -o '"prediction_label":[a-z]*' | cut -d: -f2)
  LATENCY=$(printf '%s' "${RESPONSE}" | grep -o '"latency_ms":[0-9.]*' | cut -d: -f2)
  if [[ -n "${SCORE}" && -n "${LABEL}" ]]; then
    printf "%s OK%s  (score=%.3f label=%s latency=%.1fms)\n" \
      "${GREEN}" "${RESET}" "${SCORE}" "${LABEL}" "${LATENCY:-0}"
  else
    printf "%s FAIL%s (respuesta inesperada: %s)\n" "${RED}" "${RESET}" "${RESPONSE:0:100}"
    FAILURES=$((FAILURES + 1))
  fi
fi

# ============================================================================
# Resumen final
# ============================================================================
printf "\n"
if (( FAILURES == 0 )); then
  print_success "Smoke test completado — todos los checks pasaron"
  exit 0
else
  print_error "Smoke test completado — ${FAILURES} check(s) fallaron"
  printf "  Revisá los logs con: docker compose logs <servicio>\n"
  exit 1
fi
