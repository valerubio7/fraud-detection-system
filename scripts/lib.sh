#!/usr/bin/env bash
# lib.sh — Funciones compartidas para setup.sh y deploy.sh.
# No ejecutar directamente. Importar con: source "${SCRIPT_DIR}/lib.sh"

# -------------------------------
# Color handling (TTY-aware)
# -------------------------------
if [[ -t 1 && -n "${TERM:-}" && "${TERM:-}" != "dumb" ]]; then
  if command -v tput >/dev/null 2>&1 && [[ "$(tput colors 2>/dev/null || printf '0')" -ge 8 ]]; then
    GREEN="$(tput setaf 2)"
    RED="$(tput setaf 1)"
    YELLOW="$(tput setaf 3)"
    BLUE="$(tput setaf 4)"
    RESET="$(tput sgr0)"
  else
    GREEN="" RED="" YELLOW="" BLUE="" RESET=""
  fi
else
  GREEN="" RED="" YELLOW="" BLUE="" RESET=""
fi

print_step() {
  printf "\n%s==>%s %s\n" "${BLUE}" "${RESET}" "$1"
}

print_success() {
  printf "%s✅ %s%s\n" "${GREEN}" "$1" "${RESET}"
}

print_error() {
  printf "%s❌ %s%s\n" "${RED}" "$1" "${RESET}" >&2
}

print_warning() {
  printf "%s⚠️  %s%s\n" "${YELLOW}" "$1" "${RESET}"
}

require_command() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    print_error "No se encontró el comando requerido: ${cmd}"
    exit 1
  fi
}

wait_for_service() {
  local service="$1"
  local check_fn="$2"
  local timeout="${3:-60}"
  local interval="${4:-3}"
  local elapsed=0

  print_step "Esperando a ${service} (timeout: ${timeout}s)..."
  while (( elapsed < timeout )); do
    if "${check_fn}"; then
      print_success "${service} está healthy"
      return 0
    fi
    sleep "${interval}"
    elapsed=$((elapsed + interval))
  done

  print_error "${service} no respondió en ${timeout} segundos"
  printf "   Revisá los logs con: docker compose logs %s\n" "${service}"
  return 1
}

check_postgresql() {
  docker compose exec -T postgresql pg_isready -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" >/dev/null 2>&1
}

check_timescaledb() {
  docker compose exec -T timescaledb pg_isready -U "${TIMESCALE_USER}" -d "${TIMESCALE_DB}" >/dev/null 2>&1
}

check_kafka() {
  docker compose exec -T kafka kafka-topics --bootstrap-server localhost:29092 --list >/dev/null 2>&1
}

check_mlflow() {
  curl -fsS http://localhost:5000/health >/dev/null 2>&1
}

check_serving() {
  curl -fsS http://localhost:8000/health >/dev/null 2>&1
}

check_airflow_webserver() {
  curl -fsS http://localhost:8081/health >/dev/null 2>&1
}

check_airflow_scheduler() {
  docker compose exec -T airflow-scheduler sh -lc 'airflow jobs check --job-type SchedulerJob --hostname "$HOSTNAME"' >/dev/null 2>&1
}

check_airflow_init() {
  docker compose ps --all airflow-init 2>/dev/null | grep -q "Exited (0)"
}

check_grafana() {
  curl -fsS http://localhost:3000/api/health >/dev/null 2>&1
}

check_kafka_ui() {
  curl -fsS http://localhost:8080 >/dev/null 2>&1
}

check_prometheus() {
  curl -fsS http://localhost:9090/-/healthy >/dev/null 2>&1
}

create_kafka_topic() {
  local topic="$1"
  local partitions="$2"
  local retention_ms="$3"

  docker compose exec -T kafka kafka-topics \
    --create \
    --if-not-exists \
    --topic "${topic}" \
    --bootstrap-server localhost:29092 \
    --partitions "${partitions}" \
    --replication-factor 1 \
    --config "retention.ms=${retention_ms}" >/dev/null

  print_success "Topic ${topic} listo"
}

run_sql_migrations_if_exists() {
  local service="$1"
  local db_user="$2"
  local db_name="$3"
  local label="$4"
  local migrations_dir="$5"
  local migration_files=()
  local migration_file

  if [[ ! -d "${migrations_dir}" ]]; then
    print_warning "${label}: no existe ${migrations_dir}, se omiten migraciones"
    return 0
  fi

  shopt -s nullglob
  migration_files=("${migrations_dir}"/*.sql)
  shopt -u nullglob

  if (( ${#migration_files[@]} == 0 )); then
    print_warning "${label}: no hay archivos .sql en ${migrations_dir}, se omite"
    return 0
  fi

  for migration_file in "${migration_files[@]}"; do
    print_step "${label}: ejecutando ${migration_file}"
    docker compose exec -T "${service}" psql -v ON_ERROR_STOP=1 -U "${db_user}" -d "${db_name}" -f - < "${migration_file}"
  done

  print_success "${label}: ${#migration_files[@]} migración(es) aplicada(s)"
}
