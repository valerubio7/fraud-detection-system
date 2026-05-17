#!/usr/bin/env bash
set -euo pipefail

# deploy.sh — Bootstrap desde imágenes pre-construidas en ghcr.io.
# Diferencias vs setup.sh:
#   - Usa 'docker compose pull' en lugar de 'docker compose build'.
#   - NO aplica docker-compose.override.yml (modo producción).
#   - Llama a smoke_test.sh al final para verificar el despliegue.
#
# Uso:
#   ./scripts/deploy.sh                          # último tag 'latest'
#   IMAGE_TAG=sha-a1b2c3d ./scripts/deploy.sh   # tag específico de CI

cd "$(dirname "$0")/.."

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

IMAGE_TAG="${IMAGE_TAG:-latest}"
COMPOSE_FILE="docker-compose.yml"   # sin override — producción

# ============================================================================
# Etapa 1 — Prerequisitos
# ============================================================================
print_step "Etapa 1/5 — Verificando prerequisitos"

require_command docker
require_command curl

if ! docker info >/dev/null 2>&1; then
  print_error "Docker no está corriendo"
  exit 1
fi

if [[ ! -f .env ]]; then
  if [[ ! -f .env.example ]]; then
    print_error "No existe .env ni .env.example"
    exit 1
  fi
  cp .env.example .env
  print_warning "Se creó .env desde .env.example — editá los passwords y volvé a correr deploy.sh"
  exit 0
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

required_env_vars=(
  POSTGRES_USER POSTGRES_DB
  TIMESCALE_USER TIMESCALE_DB
  KAFKA_TOPICS_RAW KAFKA_TOPICS_FEATURES
  KAFKA_TOPICS_PREDICTIONS KAFKA_TOPICS_ALERTS
  AIRFLOW_ADMIN_USER AIRFLOW_ADMIN_PASSWORD
)
for var_name in "${required_env_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    print_error "Variable no definida en .env: ${var_name}"
    exit 1
  fi
done

print_success "Prerequisitos validados (IMAGE_TAG=${IMAGE_TAG})"

# ============================================================================
# Etapa 2 — Pull de imágenes y arranque
# ============================================================================
print_step "Etapa 2/5 — Descargando imágenes desde ghcr.io (tag: ${IMAGE_TAG})"
IMAGE_TAG="${IMAGE_TAG}" docker compose -f "${COMPOSE_FILE}" pull

print_step "Etapa 2/5 — Levantando servicios base (Airflow diferido)"
IMAGE_TAG="${IMAGE_TAG}" docker compose -f "${COMPOSE_FILE}" up -d \
  --scale airflow-webserver=0 \
  --scale airflow-scheduler=0 \
  --scale airflow-init=0
print_success "Servicios base iniciados"

# ============================================================================
# Etapa 3 — Esperar servicios base
# ============================================================================
print_step "Etapa 3/5 — Esperando healthchecks"

wait_for_service "postgresql"  check_postgresql  90 3
wait_for_service "timescaledb" check_timescaledb 90 3
wait_for_service "kafka"       check_kafka       90 3
wait_for_service "mlflow"      check_mlflow      90 3

print_step "Inicializando MLflow (experimento y registro)..."
IMAGE_TAG="${IMAGE_TAG}" docker compose -f "${COMPOSE_FILE}" run --rm \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -e PYTHONPATH=/app \
  mlflow python mlops/mlflow/init_mlflow.py \
  && print_success "MLflow inicializado"

wait_for_service "fastapi"    check_fastapi    90 3
wait_for_service "prometheus" check_prometheus 60 3

# ============================================================================
# Etapa 4 — Inicializar Airflow, Kafka y migraciones
# ============================================================================
print_step "Etapa 4/5 — Inicializando Airflow, Kafka y migraciones SQL"

docker compose -f "${COMPOSE_FILE}" exec -T postgresql psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -tc \
  "SELECT 1 FROM pg_database WHERE datname = 'airflow_metadata'" \
  | grep -q 1 \
  || docker compose -f "${COMPOSE_FILE}" exec -T postgresql psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" \
       -c "CREATE DATABASE airflow_metadata;"
print_success "Base de datos airflow_metadata lista"

if ! docker compose -f "${COMPOSE_FILE}" ps --all airflow-init 2>/dev/null | grep -q "Exited (0)"; then
  IMAGE_TAG="${IMAGE_TAG}" docker compose -f "${COMPOSE_FILE}" up -d airflow-init
fi
wait_for_service "airflow-init" check_airflow_init 180 5
docker compose -f "${COMPOSE_FILE}" rm -f airflow-init >/dev/null 2>&1 || true

IMAGE_TAG="${IMAGE_TAG}" docker compose -f "${COMPOSE_FILE}" up -d \
  airflow-webserver airflow-scheduler
wait_for_service "airflow-webserver" check_airflow_webserver 120 3
wait_for_service "airflow-scheduler" check_airflow_scheduler 120 3

create_kafka_topic "${KAFKA_TOPICS_RAW}"         3 604800000
create_kafka_topic "${KAFKA_TOPICS_FEATURES}"    3 604800000
create_kafka_topic "${KAFKA_TOPICS_PREDICTIONS}" 3 604800000
create_kafka_topic "${KAFKA_TOPICS_ALERTS}"      1 2592000000

run_sql_migrations_if_exists "postgresql"  "${POSTGRES_USER}"  "${POSTGRES_DB}"  "PostgreSQL"        "database/postgresql/migrations"
run_sql_migrations_if_exists "postgresql"  "${POSTGRES_USER}"  "${POSTGRES_DB}"  "Stored procedures" "database/postgresql/stored_procedures"
run_sql_migrations_if_exists "postgresql"  "${POSTGRES_USER}"  "${POSTGRES_DB}"  "Triggers"          "database/postgresql/triggers"
run_sql_migrations_if_exists "timescaledb" "${TIMESCALE_USER}" "${TIMESCALE_DB}" "TimescaleDB"       "database/timescaledb/migrations"

# ============================================================================
# Etapa 5 — Verificación final
# ============================================================================
print_step "Etapa 5/5 — Verificación final y smoke test"

wait_for_service "grafana"   check_grafana   60 3
wait_for_service "kafka-ui"  check_kafka_ui  30 3

print_success "Stack desplegado exitosamente"

printf "\nServicios:\n"
printf "  FastAPI    → http://localhost:8000/docs\n"
printf "  MLflow     → http://localhost:5000\n"
printf "  Airflow    → http://localhost:8081  (usuario: %s)\n" "${AIRFLOW_ADMIN_USER}"
printf "  Grafana    → http://localhost:3000\n"
printf "  Prometheus → http://localhost:9090\n"
printf "  Kafka UI   → http://localhost:8080\n"

if [[ -x "${SCRIPT_DIR}/smoke_test.sh" ]]; then
  printf "\n"
  "${SCRIPT_DIR}/smoke_test.sh"
else
  print_warning "smoke_test.sh no encontrado o no ejecutable — verificación manual requerida"
fi
