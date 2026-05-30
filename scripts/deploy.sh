#!/usr/bin/env bash
set -euo pipefail

# Uso:
#   ./scripts/deploy.sh                          # tag 'latest'
#   IMAGE_TAG=sha-a1b2c3d ./scripts/deploy.sh   # tag específico de CI

cd "$(dirname "$0")/.."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

IMAGE_TAG="${IMAGE_TAG:-latest}"
COMPOSE_FILE="docker-compose.yml"

# ============================================================================
# Etapa 1/6 — Prerequisitos
# ============================================================================
print_step "Etapa 1/6 — Verificando prerequisitos (Docker, .env, variables)"

require_command docker
require_command curl

if ! docker info >/dev/null 2>&1; then
  print_error "Docker no está corriendo"
  exit 1
fi

if [[ ! -f .env ]]; then
  if [[ ! -f .env.example ]]; then
    print_error "No existe .env ni .env.example en la raíz del proyecto"
    exit 1
  fi
  cp .env.example .env
  print_warning "Se creó .env desde .env.example — editá los passwords antes de continuar y volvé a correr deploy.sh"
  exit 0
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

required_env_vars=(
  AIRFLOW_ADMIN_USER AIRFLOW_ADMIN_PASSWORD
  POSTGRES_USER POSTGRES_DB
  TIMESCALE_USER TIMESCALE_DB
  KAFKA_TOPICS_RAW KAFKA_TOPICS_FEATURES
  KAFKA_TOPICS_PREDICTIONS KAFKA_TOPICS_ALERTS
)

for var_name in "${required_env_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    print_error "Variable no definida en .env: ${var_name}"
    exit 1
  fi
done

print_success "Prerequisitos validados (IMAGE_TAG=${IMAGE_TAG})"

# ============================================================================
# Etapa 2/6 — Pull de imágenes y arranque del stack
# ============================================================================
print_step "Etapa 2/6 — Descargando imágenes desde el registry (tag: ${IMAGE_TAG})"
IMAGE_TAG="${IMAGE_TAG}" docker compose -f "${COMPOSE_FILE}" pull

print_step "Levantando el stack base (Airflow se inicializa por separado en la etapa 4)"
IMAGE_TAG="${IMAGE_TAG}" docker compose -f "${COMPOSE_FILE}" up -d \
  --scale airflow-webserver=0 \
  --scale airflow-scheduler=0 \
  --scale airflow-init=0
print_success "Stack base levantado"

# ============================================================================
# Etapa 3/6 — Servicios de infraestructura y MLflow
# ============================================================================
print_step "Etapa 3/6 — Esperando servicios de infraestructura e inicializando MLflow"

wait_for_service "postgresql"  check_postgresql  90 3
wait_for_service "timescaledb" check_timescaledb 90 3
wait_for_service "kafka"       check_kafka       90 3
wait_for_service "mlflow"      check_mlflow      90 3

print_step "Inicializando experimento y model registry en MLflow..."
IMAGE_TAG="${IMAGE_TAG}" docker compose -f "${COMPOSE_FILE}" run --rm \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -e PYTHONPATH=/app \
  mlflow python mlops/mlflow/init_mlflow.py
print_success "MLflow inicializado (experimento: fraud-detection-v1, registry: FraudDetectionModel)"

wait_for_service "prometheus" check_prometheus 60 3

# ============================================================================
# Etapa 4/6 — Airflow, Kafka y migraciones SQL
# ============================================================================
print_step "Etapa 4/6 — Inicializando Airflow, creando topics Kafka y aplicando migraciones SQL"

print_step "Creando base de datos airflow_metadata en PostgreSQL si no existe..."
docker compose -f "${COMPOSE_FILE}" exec -T postgresql psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -tc \
  "SELECT 1 FROM pg_database WHERE datname = 'airflow_metadata'" | grep -q 1 \
  || docker compose -f "${COMPOSE_FILE}" exec -T postgresql psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" \
       -c "CREATE DATABASE airflow_metadata;"
print_success "Base de datos airflow_metadata lista"

print_step "Ejecutando airflow-init (migraciones de BD + creación de usuario admin)..."
if ! docker compose -f "${COMPOSE_FILE}" ps --all airflow-init 2>/dev/null | grep -q "Exited (0)"; then
  IMAGE_TAG="${IMAGE_TAG}" docker compose -f "${COMPOSE_FILE}" up -d airflow-init
fi
wait_for_service "airflow-init" check_airflow_init 180 5
docker compose -f "${COMPOSE_FILE}" rm -f airflow-init >/dev/null 2>&1 || true

print_step "Levantando Airflow webserver y scheduler..."
IMAGE_TAG="${IMAGE_TAG}" docker compose -f "${COMPOSE_FILE}" up -d airflow-webserver airflow-scheduler
wait_for_service "airflow-webserver" check_airflow_webserver 120 3
wait_for_service "airflow-scheduler"  check_airflow_scheduler  120 3

print_step "Creando topics Kafka con retención y particiones configuradas..."
create_kafka_topic "${KAFKA_TOPICS_RAW}"         3 604800000
create_kafka_topic "${KAFKA_TOPICS_FEATURES}"    3 604800000
create_kafka_topic "${KAFKA_TOPICS_PREDICTIONS}" 3 604800000
create_kafka_topic "${KAFKA_TOPICS_ALERTS}"      1 2592000000

run_sql_migrations_if_exists "postgresql"  "${POSTGRES_USER}"  "${POSTGRES_DB}"  "PostgreSQL migrations"  "database/postgresql/migrations"
run_sql_migrations_if_exists "postgresql"  "${POSTGRES_USER}"  "${POSTGRES_DB}"  "PostgreSQL procedures"  "database/postgresql/stored_procedures"
run_sql_migrations_if_exists "postgresql"  "${POSTGRES_USER}"  "${POSTGRES_DB}"  "PostgreSQL triggers"    "database/postgresql/triggers"
run_sql_migrations_if_exists "timescaledb" "${TIMESCALE_USER}" "${TIMESCALE_DB}" "TimescaleDB migrations" "database/timescaledb/migrations"

# ============================================================================
# Etapa 5/6 — Seed, entrenamiento y modelo en producción (idempotente por paso)
# ============================================================================
print_step "Etapa 5/6 — Seed, entrenamiento y modelo en producción (idempotente por paso)"

_mlflow_version() {
  local stage="$1"
  curl -sS \
    "http://localhost:5000/api/2.0/mlflow/registered-models/get-latest-versions?name=FraudDetectionModel&stages=${stage}" | \
    python3 -c "
import sys, json
d = json.load(sys.stdin)
v = d.get('model_versions', [])
print(v[0]['version'] if v else '')
" 2>/dev/null | tr -d '[:space:]'
}

PROD_VERSION=$(_mlflow_version "Production")

if [[ -n "${PROD_VERSION}" ]]; then
  print_warning "Modelo v${PROD_VERSION} ya está en Production — se omiten seed, entrenamiento y promoción"
  MODEL_VERSION="${PROD_VERSION}"
  IMAGE_TAG="${IMAGE_TAG}" docker compose -f "${COMPOSE_FILE}" up -d serving
  wait_for_service "serving" check_serving 60 3
else
  # ── Seed ─────────────────────────────────────────────────────────────────
  TX_COUNT=$(docker compose -f "${COMPOSE_FILE}" exec -T timescaledb psql \
    -U "${TIMESCALE_USER}" -d "${TIMESCALE_DB}" \
    -tAc "SELECT COUNT(*) FROM public.transactions;" 2>/dev/null | tr -d '[:space:]')

  if [[ "${TX_COUNT:-0}" -ge "50000" ]]; then
    print_warning "TimescaleDB ya tiene ${TX_COUNT} transacciones — se omite el seed"
  else
    print_step "Generando 100.000 transacciones sintéticas (fraud rate 2%) en TimescaleDB..."
    print_warning "Esto puede tardar unos minutos..."
    docker compose -f "${COMPOSE_FILE}" exec -T -w /opt/airflow/project airflow-scheduler \
      python database/timescaledb/seeds/seed_transactions.py --count 100000 --fraud-rate 0.02
    print_success "100.000 transacciones insertadas en TimescaleDB"
  fi

  # ── Entrenamiento ─────────────────────────────────────────────────────────
  STAGING_VERSION=$(_mlflow_version "Staging")

  if [[ -n "${STAGING_VERSION}" ]]; then
    print_warning "Modelo v${STAGING_VERSION} ya está en Staging — se omite el entrenamiento"
  else
    print_step "Entrenando modelo XGBoost de detección de fraude..."
    print_warning "Incluye feature engineering y logging a MLflow. Puede tardar varios minutos..."
    docker compose -f "${COMPOSE_FILE}" exec -T -w /opt/airflow/project airflow-scheduler \
      python model/pipeline/train.py --output-dir /tmp/fraud_model
    print_success "Entrenamiento completado — modelo registrado en MLflow (stage: Staging)"

    STAGING_VERSION=$(_mlflow_version "Staging")
    if [[ -z "${STAGING_VERSION}" ]]; then
      print_error "No se encontró el modelo en Staging tras el entrenamiento. Revisá los logs."
      exit 1
    fi
  fi

  # ── Quality gates + promoción ─────────────────────────────────────────────
  print_step "Ejecutando quality gates del modelo v${STAGING_VERSION} (F1 ≥ 0.85, AUC-ROC ≥ 0.90, latencia P99 ≤ 50ms)..."
  print_warning "Primer modelo: se promueve a Production independientemente del resultado."
  docker compose -f "${COMPOSE_FILE}" exec -T -w /opt/airflow/project airflow-scheduler \
    python model/pipeline/evaluate.py \
    --model-name FraudDetectionModel \
    --model-version "${STAGING_VERSION}" || true

  print_step "Promoviendo modelo v${STAGING_VERSION} a Production..."
  docker compose -f "${COMPOSE_FILE}" exec -T -w /opt/airflow/project airflow-scheduler \
    python model/pipeline/promote.py \
    --model-name FraudDetectionModel \
    --model-version "${STAGING_VERSION}"
  print_success "Modelo v${STAGING_VERSION} promovido a Production"
  MODEL_VERSION="${STAGING_VERSION}"

  print_step "Reiniciando serving para cargar el modelo v${MODEL_VERSION}..."
  docker compose -f "${COMPOSE_FILE}" restart serving
  wait_for_service "serving" check_serving 60 3
  print_success "Serving activo con el modelo v${MODEL_VERSION} en producción"
fi

# ============================================================================
# Etapa 6/6 — Smoke test y resumen
# ============================================================================
print_step "Etapa 6/6 — Verificación final y resumen del despliegue"

wait_for_service "postgresql"        check_postgresql        30 3
wait_for_service "timescaledb"       check_timescaledb       30 3
wait_for_service "kafka"             check_kafka             30 3
wait_for_service "mlflow"            check_mlflow            30 3
wait_for_service "serving"           check_serving           30 3
wait_for_service "airflow-webserver" check_airflow_webserver 30 3
wait_for_service "prometheus"        check_prometheus        30 3
wait_for_service "grafana"           check_grafana           30 3

print_success "Despliegue completado exitosamente"

printf "\n"
printf "Servicios en producción:\n"
printf "  %-18s http://localhost:8000      API REST de predicción de fraude\n"      "serving"
printf "  %-18s http://localhost:8000/docs Documentación interactiva (Swagger UI)\n" "serving/docs"
printf "  %-18s http://localhost:5000      Experimentos, métricas y model registry\n" "mlflow"
printf "  %-18s http://localhost:8081      Orquestación de pipelines de reentrenamiento\n" "airflow"
printf "  %-18s http://localhost:9090      Métricas del sistema y alertas\n"         "prometheus"
printf "  %-18s http://localhost:3000      Dashboards de monitoreo en tiempo real\n" "grafana"

printf "\n"
printf "Modelo en producción:\n"
printf "  Nombre:  FraudDetectionModel\n"
printf "  Versión: v%s\n" "${MODEL_VERSION}"

printf "\n"
printf "Comandos útiles:\n"
printf "  Ver logs de un servicio:  docker compose -f docker-compose.yml logs -f <servicio>\n"
printf "  Detener el stack:         docker compose -f docker-compose.yml down\n"
printf "  Re-desplegar (mismo tag): IMAGE_TAG=%s ./scripts/deploy.sh\n" "${IMAGE_TAG}"
