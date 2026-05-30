#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

# ============================================================================
# Etapa 1/6 — Prerequisitos
# ============================================================================
print_step "Etapa 1/6 — Verificando prerequisitos (Docker, .env, variables)"

require_command docker
require_command curl

if ! docker info >/dev/null 2>&1; then
  print_error "Docker está instalado pero no está corriendo"
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  print_error "Docker Compose v2 no está disponible. Usá 'docker compose' (no docker-compose v1)."
  exit 1
fi

compose_version="$(docker compose version --short 2>/dev/null || true)"
if [[ -n "${compose_version}" ]]; then
  compose_major="${compose_version#v}"
  compose_major="${compose_major%%.*}"
  if ! [[ "${compose_major}" =~ ^[0-9]+$ ]] || (( compose_major < 2 )); then
    print_error "Se requiere Docker Compose v2 (detectado: ${compose_version})"
    exit 1
  fi
fi

if [[ ! -f .env ]]; then
  if [[ ! -f .env.example ]]; then
    print_error "No existe .env ni .env.example en la raíz del proyecto"
    exit 1
  fi
  cp .env.example .env
  print_warning "Se creó .env desde .env.example — editá los passwords antes de continuar y volvé a correr ./scripts/setup.sh"
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

print_success "Prerequisitos validados"

# ============================================================================
# Etapa 2/6 — Build de imágenes y arranque del stack
# ============================================================================
print_step "Etapa 2/6 — Construyendo imágenes Docker locales"
print_warning "Este paso puede tardar varios minutos la primera vez..."
docker compose build

print_step "Levantando el stack base (Airflow se inicializa por separado en la etapa 4)"
docker compose up -d \
  --scale airflow-webserver=0 \
  --scale airflow-scheduler=0 \
  --scale airflow-init=0
print_success "Stack base levantado"

# ============================================================================
# Etapa 3/6 — Servicios de infraestructura y MLflow
# ============================================================================
print_step "Etapa 3/6 — Esperando servicios de infraestructura e inicializando MLflow"

wait_for_service "postgresql"  check_postgresql  60 3
wait_for_service "timescaledb" check_timescaledb 60 3
wait_for_service "kafka"       check_kafka       60 3
wait_for_service "mlflow"      check_mlflow      60 3

print_step "Inicializando experimento y model registry en MLflow..."
docker compose run --rm \
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
docker compose exec -T postgresql psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -tc \
  "SELECT 1 FROM pg_database WHERE datname = 'airflow_metadata'" | grep -q 1 \
  || docker compose exec -T postgresql psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" \
       -c "CREATE DATABASE airflow_metadata;"
print_success "Base de datos airflow_metadata lista"

print_step "Ejecutando airflow-init (migraciones de BD + creación de usuario admin)..."
if ! docker compose ps --all airflow-init 2>/dev/null | grep -q "Exited (0)"; then
  docker compose up -d airflow-init
fi
wait_for_service "airflow-init" check_airflow_init 120 5
docker compose rm -f airflow-init >/dev/null 2>&1 || true

print_step "Levantando Airflow webserver y scheduler..."
docker compose up -d airflow-webserver airflow-scheduler
wait_for_service "airflow-webserver" check_airflow_webserver 90 3
wait_for_service "airflow-scheduler"  check_airflow_scheduler  90 3

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
  docker compose up -d serving
  wait_for_service "serving" check_serving 60 3
else
  # ── Seed ─────────────────────────────────────────────────────────────────
  TX_COUNT=$(docker compose exec -T timescaledb psql \
    -U "${TIMESCALE_USER}" -d "${TIMESCALE_DB}" \
    -tAc "SELECT COUNT(*) FROM public.transactions;" 2>/dev/null | tr -d '[:space:]')

  if [[ "${TX_COUNT:-0}" -ge "50000" ]]; then
    print_warning "TimescaleDB ya tiene ${TX_COUNT} transacciones — se omite el seed"
  else
    print_step "Generando 100.000 transacciones sintéticas (fraud rate 2%) en TimescaleDB..."
    print_warning "Esto puede tardar unos minutos..."
    docker compose exec -T -w /opt/airflow/project airflow-scheduler \
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
    docker compose exec -T -w /opt/airflow/project airflow-scheduler \
      python model/pipeline/train.py --output-dir /tmp/fraud_model
    print_success "Entrenamiento completado — modelo registrado en MLflow (stage: Staging)"

    if FEATURES_JSON=$(docker compose exec -T airflow-scheduler bash -c "cat /tmp/selected_features.json 2>/dev/null") && [[ -n "${FEATURES_JSON}" ]]; then
      python3 -c "
import json, sys
features = json.loads(sys.stdin.read())
body = '\n'.join(f'    \"{f}\",' for f in features)
content = '''\"\"\"
Lista definitiva de features para el FraudDetectionModel.
Auto-actualizada por setup.sh tras la seleccion de features sobre el dataset de entrenamiento.
\"\"\"

SELECTED_FEATURES: list[str] = [
''' + body + '''
]
'''
with open('model/utils/selected_features.py', 'w') as fh:
    fh.write(content)
print(f'{len(features)} features escritas en selected_features.py')
" <<< "${FEATURES_JSON}"
      print_success "model/utils/selected_features.py actualizado automáticamente"
      print_step "Reconstruyendo serving con las features actualizadas..."
      docker compose build serving
      print_success "Imagen de serving reconstruida"
    fi

    STAGING_VERSION=$(_mlflow_version "Staging")
    if [[ -z "${STAGING_VERSION}" ]]; then
      print_error "No se encontró el modelo en Staging tras el entrenamiento. Revisá los logs."
      exit 1
    fi
  fi

  # ── Quality gates + promoción ─────────────────────────────────────────────
  print_step "Ejecutando quality gates del modelo v${STAGING_VERSION} (F1 ≥ 0.85, AUC-ROC ≥ 0.90, latencia P99 ≤ 50ms)..."
  print_warning "Primer modelo: se promueve a Production independientemente del resultado."
  docker compose exec -T -w /opt/airflow/project airflow-scheduler \
    python model/pipeline/evaluate.py \
    --model-name FraudDetectionModel \
    --model-version "${STAGING_VERSION}" || true

  print_step "Promoviendo modelo v${STAGING_VERSION} a Production..."
  docker compose exec -T -w /opt/airflow/project airflow-scheduler \
    python model/pipeline/promote.py \
    --model-name FraudDetectionModel \
    --model-version "${STAGING_VERSION}"
  print_success "Modelo v${STAGING_VERSION} promovido a Production"
  MODEL_VERSION="${STAGING_VERSION}"

  print_step "Reiniciando serving para cargar el modelo v${MODEL_VERSION}..."
  docker compose restart serving
  wait_for_service "serving" check_serving 60 3
  print_success "Serving activo con el modelo v${MODEL_VERSION} en producción"
fi

# ============================================================================
# Etapa 6/6 — Verificación final y resumen
# ============================================================================
print_step "Etapa 6/6 — Verificación final de todos los servicios"

wait_for_service "postgresql"        check_postgresql        30 3
wait_for_service "timescaledb"       check_timescaledb       30 3
wait_for_service "kafka"             check_kafka             30 3
wait_for_service "mlflow"            check_mlflow            30 3
wait_for_service "serving"           check_serving           30 3
wait_for_service "airflow-webserver" check_airflow_webserver 30 3
wait_for_service "prometheus"        check_prometheus        30 3
wait_for_service "grafana"           check_grafana           30 3
wait_for_service "kafka-ui"          check_kafka_ui          30 3

print_success "Setup completado — todos los servicios están healthy"

printf "\n"
printf "Servicios disponibles:\n"
printf "  %-18s http://localhost:8000      API REST de predicción de fraude en tiempo real\n"  "serving"
printf "  %-18s http://localhost:8000/docs Documentación interactiva de la API (Swagger UI)\n" "serving/docs"
printf "  %-18s http://localhost:5000      Experimentos, métricas y model registry\n"          "mlflow"
printf "  %-18s http://localhost:8081      Orquestación de pipelines de reentrenamiento\n"     "airflow"
printf "  %-18s http://localhost:9090      Métricas del sistema y alertas\n"                   "prometheus"
printf "  %-18s http://localhost:3000      Dashboards de monitoreo en tiempo real\n"           "grafana"
printf "  %-18s http://localhost:8080      Interfaz visual de Kafka (solo disponible en dev)\n" "kafka-ui"

printf "\n"
printf "Modelo en producción:\n"
printf "  Nombre:   FraudDetectionModel\n"
printf "  Versión:  v%s\n" "${MODEL_VERSION}"
printf "  MLflow:   http://localhost:5000/#/models/FraudDetectionModel\n"

printf "\n"
printf "Comandos útiles:\n"
printf "  Ver logs de un servicio:       docker compose logs -f <servicio>\n"
printf "  Detener el stack:              docker compose down\n"
printf "  Reentrenar modelo manualmente: ./scripts/setup.sh (re-seed + retrain)\n"
