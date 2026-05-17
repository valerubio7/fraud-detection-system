#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Fraud Detection MLOps - Initial local setup script
#
# This script is intentionally idempotent:
# - it can be run multiple times safely
# - resources that already exist are reused (not recreated destructively)
# -----------------------------------------------------------------------------

# Always run from the project root, even when invoked elsewhere.
cd "$(dirname "$0")/.."

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

print_step "Contexto del setup"
printf "Este script prepara el entorno local con Docker Compose y es idempotente.\n"
printf "Hace lo siguiente:\n"
printf "  - valida Docker/Compose y crea .env desde .env.example si falta\n"
printf "  - construye imágenes locales (FastAPI, Airflow, MLflow,\n"
printf "    Kafka producer, Grafana)\n"
printf "  - levanta servicios base y espera healthchecks\n"
printf "  - inicializa Airflow, topics Kafka y migraciones SQL\n"
printf "  - verifica salud y muestra URLs/comandos útiles\n"
printf "\nNo hace por diseño:\n"
printf "  - no ejecuta notebooks ni pipelines locales de Python\n"
printf "  - no genera datos históricos salvo que lo ejecutes manualmente\n"
printf "\nNota: docker-compose.override.yml se aplica automáticamente para dev.\n"

# ============================================================================
# Etapa 1 — Verificar prerequisitos
# ============================================================================
print_step "Etapa 1/5 — Verificando prerequisitos (Docker/Compose + .env)"

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
    print_error "Se detectó una versión no compatible de Docker Compose: ${compose_version}"
    print_error "Se requiere Docker Compose v2 o superior mediante 'docker compose'."
    exit 1
  fi
fi

if [[ ! -f .env ]]; then
  if [[ ! -f .env.example ]]; then
    print_error "No existe .env ni .env.example en la raíz del proyecto"
    exit 1
  fi

  cp .env.example .env
  printf "⚠️  Se creó .env desde .env.example\n"
  printf "    Editá las variables antes de continuar (especialmente passwords)\n"
  printf "    Cuando estés listo, volvé a correr ./scripts/setup.sh\n"
  exit 0
fi

# Export .env variables so subprocesses (docker compose exec commands) can use them.
set -a
# shellcheck disable=SC1091
source .env
set +a

required_env_vars=(
  AIRFLOW_ADMIN_USER
  AIRFLOW_ADMIN_PASSWORD
  POSTGRES_USER
  POSTGRES_DB
  TIMESCALE_USER
  TIMESCALE_DB
  KAFKA_TOPICS_RAW
  KAFKA_TOPICS_FEATURES
  KAFKA_TOPICS_PREDICTIONS
  KAFKA_TOPICS_ALERTS
)

for var_name in "${required_env_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    print_error "La variable ${var_name} no está definida en .env"
    exit 1
  fi
done

print_success "Prerequisitos validados"

# ============================================================================
# Etapa 2 — Construir y levantar el stack
# ============================================================================
print_step "Etapa 2/5 — Construyendo imágenes locales"
docker compose build

print_step "Etapa 2/5 — Levantando servicios base (Airflow inicia luego)"
docker compose up -d --scale airflow-webserver=0 --scale airflow-scheduler=0 --scale airflow-init=0
print_success "Stack levantado"

# ============================================================================
# Etapa 3 — Esperar que los servicios estén healthy
# ============================================================================
print_step "Etapa 3/5 — Esperando healthchecks de servicios base"

wait_for_service "postgresql" check_postgresql 60 3
wait_for_service "timescaledb" check_timescaledb 60 3
wait_for_service "kafka" check_kafka 60 3
wait_for_service "mlflow" check_mlflow 60 3

print_step "Inicializando MLflow (experimento y registry)..."
docker compose run --rm -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -e PYTHONPATH=/app \
  mlflow python mlops/mlflow/init_mlflow.py \
  && print_success "MLflow inicializado correctamente"

wait_for_service "fastapi" check_fastapi 60 3
wait_for_service "prometheus" check_prometheus 60 3

# ============================================================================
# Etapa 4 — Inicializar servicios
# ============================================================================
print_step "Etapa 4/5 — Inicializando Airflow, Kafka y migraciones SQL"

print_step "Airflow: creando base de datos airflow_metadata si no existe"
docker compose exec -T postgresql psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -tc \
  "SELECT 1 FROM pg_database WHERE datname = 'airflow_metadata'" \
  | grep -q 1 \
  || docker compose exec -T postgresql psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" \
       -c "CREATE DATABASE airflow_metadata;"
print_success "Base de datos airflow_metadata lista"

print_step "Airflow: esperando que airflow-init complete (crea BD, migra esquema, crea admin)"
if ! docker compose ps --all airflow-init 2>/dev/null | grep -q "Exited (0)"; then
  docker compose up -d airflow-init
fi
wait_for_service "airflow-init" check_airflow_init 120 5
docker compose rm -f airflow-init >/dev/null 2>&1 || true

print_step "Airflow: levantando webserver y scheduler"
docker compose up -d airflow-webserver airflow-scheduler
wait_for_service "airflow-webserver" check_airflow_webserver 90 3
wait_for_service "airflow-scheduler" check_airflow_scheduler 90 3

print_step "Kafka: creando topics base"
# transactions.raw — particiones: 3, retención: 7 días
create_kafka_topic "${KAFKA_TOPICS_RAW}" 3 604800000
# transactions.features — particiones: 3, retención: 7 días
create_kafka_topic "${KAFKA_TOPICS_FEATURES}" 3 604800000
# transactions.predictions — particiones: 3, retención: 7 días
create_kafka_topic "${KAFKA_TOPICS_PREDICTIONS}" 3 604800000
# transactions.fraud.alerts — particiones: 1, retención: 30 días
create_kafka_topic "${KAFKA_TOPICS_ALERTS}" 1 2592000000

run_sql_migrations_if_exists "postgresql" "${POSTGRES_USER}" "${POSTGRES_DB}" "PostgreSQL" "database/postgresql/migrations"
run_sql_migrations_if_exists "postgresql" "${POSTGRES_USER}" "${POSTGRES_DB}" "PostgreSQL stored procedures" "database/postgresql/stored_procedures"
run_sql_migrations_if_exists "postgresql" "${POSTGRES_USER}" "${POSTGRES_DB}" "PostgreSQL triggers" "database/postgresql/triggers"
run_sql_migrations_if_exists "timescaledb" "${TIMESCALE_USER}" "${TIMESCALE_DB}" "TimescaleDB" "database/timescaledb/migrations"

# ============================================================================
# Etapa 5 — Verificar y mostrar resumen
# ============================================================================
print_step "Etapa 5/5 — Verificación final y resumen"

wait_for_service "postgresql" check_postgresql 30 3
wait_for_service "timescaledb" check_timescaledb 30 3
wait_for_service "kafka" check_kafka 30 3
wait_for_service "mlflow" check_mlflow 30 3
wait_for_service "fastapi" check_fastapi 30 3
wait_for_service "airflow-webserver" check_airflow_webserver 30 3
wait_for_service "prometheus" check_prometheus 30 3
wait_for_service "grafana" check_grafana 30 3
wait_for_service "kafka-ui" check_kafka_ui 30 3

printf "\n"
print_success "Setup completado exitosamente"

printf "\nServicios disponibles:\n"
printf "  FastAPI       → http://localhost:8000\n"
printf "  FastAPI docs  → http://localhost:8000/docs\n"
printf "  MLflow        → http://localhost:5000\n"
printf "  Airflow       → http://localhost:8081\n"
printf "  Prometheus    → http://localhost:9090\n"
printf "  Grafana       → http://localhost:3000\n"
printf "  Kafka UI      → http://localhost:8080\n"

printf "\nComandos útiles:\n"
printf "  Ver logs:       docker compose logs -f [servicio]\n"
printf "  Detener stack:  docker compose down\n"
printf "  Producción:     docker compose -f docker-compose.yml up -d\n"
printf "  Seed data:      ver database/timescaledb/seeds/README.md\n"

printf "\nSimulaciones (producer):\n"
printf "  uv run python -m ingestion.producer.main --mode live --tps 10 --fraud-rate 0.02\n"
printf "  uv run python -m ingestion.producer.main --mode scenario --scenario high_frequency --tps 5\n"
printf "  uv run python -m ingestion.producer.main --mode replay --replay /ruta/al/archivo.csv\n"


printf "\nNotebooks (Jupyter):\n"
printf "  1) Instalar deps: uv sync --group notebooks\n"
printf "  2) Iniciar:       uv run jupyter lab\n"
printf "  3) Abrir:         model/notebooks/eda_base.ipynb\n"
