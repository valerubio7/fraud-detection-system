#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

# ── defaults ──────────────────────────────────────────────────────────────────
MODE="live"
SCENARIO="amount_anomaly"
TPS=10
FRAUD_RATE="0.02"
NUM_USERS=200
NUM_MERCHANTS=50
DURATION=0

# ── helpers ───────────────────────────────────────────────────────────────────
fraud_rate_pct() { python3 -c "print(f\"{float('${FRAUD_RATE}') * 100:.0f}%\")"; }
duration_label() { [[ "${DURATION}" -eq 0 ]] && printf "infinite" || printf "%ss" "${DURATION}"; }

show_config() {
  printf "\n%sCurrent configuration:%s\n" "${BLUE}" "${RESET}"
  printf "  Mode:        %s\n" "${MODE}"
  if [[ "${MODE}" == "scenario" ]]; then
    printf "  Scenario:    %s\n" "${SCENARIO}"
  else
    printf "  Fraud rate:  %s\n" "$(fraud_rate_pct)"
  fi
  printf "  TPS:         %s tx/sec\n" "${TPS}"
  printf "  Users:       %s\n" "${NUM_USERS}"
  printf "  Merchants:   %s\n" "${NUM_MERCHANTS}"
  printf "  Duration:    %s\n" "$(duration_label)"
  printf "\n"
}

show_menu() {
  show_config
  printf "  [1] Mode       (%s)\n"     "${MODE}"
  if [[ "${MODE}" == "scenario" ]]; then
    printf "  [2] Scenario   (%s)\n"   "${SCENARIO}"
  else
    printf "  [2] Fraud rate (%s)\n"   "$(fraud_rate_pct)"
  fi
  printf "  [3] TPS        (%s tx/sec)\n" "${TPS}"
  printf "  [4] Duration   (%s)\n"    "$(duration_label)"
  printf "  [5] Users      (%s)\n"    "${NUM_USERS}"
  printf "  [6] Merchants  (%s)\n"    "${NUM_MERCHANTS}"
  printf "\n"
  printf "  [s] Start\n"
  printf "  [q] Quit\n"
  printf "\n"
}

prompt_mode() {
  printf "\nSelect mode:\n"
  printf "  [1] live     — continuous stream with configurable fraud rate\n"
  printf "  [2] scenario — inject a specific fraud pattern\n"
  printf "Choice [1/2]: "
  read -r choice
  case "${choice}" in
    1) MODE="live" ;;
    2) MODE="scenario" ;;
    *) print_warning "Invalid choice, keeping '${MODE}'" ;;
  esac
}

prompt_scenario() {
  printf "\nSelect fraud scenario:\n"
  printf "  [1] amount_anomaly   — charge 5-10x the user's normal amount\n"
  printf "  [2] unusual_country  — transaction from geographically distant country\n"
  printf "  [3] high_frequency   — 5-8 rapid transactions within 30 minutes\n"
  printf "  [4] unknown_merchant — high-value purchase at a new merchant\n"
  printf "Choice [1-4]: "
  read -r choice
  case "${choice}" in
    1) SCENARIO="amount_anomaly" ;;
    2) SCENARIO="unusual_country" ;;
    3) SCENARIO="high_frequency" ;;
    4) SCENARIO="unknown_merchant" ;;
    *) print_warning "Invalid choice, keeping '${SCENARIO}'" ;;
  esac
}

prompt_fraud_rate() {
  printf "\nFraud rate as percentage (current: %s, e.g. 5 for 5%%): " "$(fraud_rate_pct)"
  read -r val
  if [[ "${val}" =~ ^[0-9]+(\.[0-9]+)?$ ]] && python3 -c "exit(0 if 0 <= float('${val}') <= 100 else 1)" 2>/dev/null; then
    FRAUD_RATE=$(python3 -c "print(float('${val}') / 100)")
  else
    print_warning "Invalid value (0–100), keeping $(fraud_rate_pct)"
  fi
}

prompt_tps() {
  printf "\nTransactions per second (current: %s, range 1–500): " "${TPS}"
  read -r val
  if [[ "${val}" =~ ^[0-9]+$ ]] && (( val >= 1 && val <= 500 )); then
    TPS="${val}"
  else
    print_warning "Invalid value, keeping ${TPS}"
  fi
}

prompt_duration() {
  printf "\nDuration in seconds (current: %s, 0 = infinite): " "$(duration_label)"
  read -r val
  if [[ "${val}" =~ ^[0-9]+$ ]]; then
    DURATION="${val}"
  else
    print_warning "Invalid value, keeping $(duration_label)"
  fi
}

prompt_users() {
  printf "\nNumber of simulated users (current: %s): " "${NUM_USERS}"
  read -r val
  if [[ "${val}" =~ ^[0-9]+$ ]] && (( val >= 1 )); then
    NUM_USERS="${val}"
  else
    print_warning "Invalid value, keeping ${NUM_USERS}"
  fi
}

prompt_merchants() {
  printf "\nNumber of simulated merchants (current: %s): " "${NUM_MERCHANTS}"
  read -r val
  if [[ "${val}" =~ ^[0-9]+$ ]] && (( val >= 1 )); then
    NUM_MERCHANTS="${val}"
  else
    print_warning "Invalid value, keeping ${NUM_MERCHANTS}"
  fi
}

_RESTORE=false

run_streaming() {
  local -a cmd=(
    python -m streaming.producer.main
    --mode  "${MODE}"
    --tps   "${TPS}"
    --num-users     "${NUM_USERS}"
    --num-merchants "${NUM_MERCHANTS}"
  )

  if [[ "${MODE}" == "live" ]]; then
    cmd+=(--fraud-rate "${FRAUD_RATE}")
  else
    cmd+=(--scenario "${SCENARIO}")
  fi

  if (( DURATION > 0 )); then
    cmd+=(--duration "${DURATION}")
  fi

  show_config
  printf "  Running: %s\n\n" "${cmd[*]}"

  _RESTORE=false
  if docker compose ps producer 2>/dev/null | grep -qE "running|Up"; then
    print_warning "Pausing default producer..."
    docker compose stop producer
    _RESTORE=true
  fi

  local container_id
  container_id=$(docker compose run -d producer "${cmd[@]}")

  cleanup() {
    printf "\n"
    print_step "Stopping custom producer..."
    docker stop "${container_id}" 2>/dev/null || true
    docker rm   "${container_id}" 2>/dev/null || true
    if [[ "${_RESTORE}" == true ]]; then
      print_step "Restoring default producer..."
      docker compose start producer
    fi
  }
  trap cleanup EXIT INT TERM

  print_step "Custom producer started (id: ${container_id:0:12}) — press Enter to stop"
  read -r _ignored || true
  cleanup
  trap - EXIT INT TERM
}

# ── main ──────────────────────────────────────────────────────────────────────
print_step "Fraud Detection System — Streaming Control"

while true; do
  show_menu
  printf "Choice: "
  read -r choice
  case "${choice}" in
    1)   prompt_mode ;;
    2)   [[ "${MODE}" == "scenario" ]] && prompt_scenario || prompt_fraud_rate ;;
    3)   prompt_tps ;;
    4)   prompt_duration ;;
    5)   prompt_users ;;
    6)   prompt_merchants ;;
    s|S) run_streaming; break ;;
    q|Q) exit 0 ;;
    *)   print_warning "Invalid option" ;;
  esac
done
