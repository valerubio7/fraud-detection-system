# Runbook: Reiniciar servicios individualmente

## Cuándo usarlo
- Un servicio está en estado `unhealthy` o `Exited`.
- Se desplegó una nueva imagen y se necesita aplicarla sin bajar todo el stack.
- Un servicio consume recursos anómalos y necesita reinicio.

## Diagnóstico previo

```bash
docker compose ps                          # estado de todos los servicios
docker compose logs --tail=50 <servicio>   # últimas líneas del log
```

## Reinicio simple (conserva datos)

```bash
docker compose restart <servicio>
```

## Reinicio completo (recrea el contenedor)

```bash
docker compose up -d --force-recreate <servicio>
```

## Reinicio con imagen actualizada

```bash
docker compose pull <servicio>             # descarga la nueva imagen
docker compose up -d --force-recreate <servicio>
```

## Servicios y sus dependencias

| Servicio | Dependencias críticas | Tiempo de arranque |
|---|---|---|
| `fastapi` | postgresql, timescaledb, mlflow | ~30s (carga modelo) |
| `consumer` | kafka, timescaledb, redis | ~10s |
| `inference_consumer` | kafka, fastapi | ~10s |
| `airflow-webserver` | postgresql (airflow_metadata) | ~60s |
| `airflow-scheduler` | postgresql (airflow_metadata) | ~30s |
| `mlflow` | postgresql | ~15s |
| `grafana` | postgresql, prometheus | ~20s |

## Reiniciar Airflow (webserver + scheduler juntos)

```bash
docker compose restart airflow-webserver airflow-scheduler
# Verificar:
curl -fsS http://localhost:8081/health
```

## Verificar estado post-reinicio

```bash
docker compose ps <servicio>
# El campo STATUS debe mostrar "(healthy)" en menos de 2 minutos.
# Si queda en "(starting)", revisar logs:
docker compose logs --tail=100 <servicio>
```
