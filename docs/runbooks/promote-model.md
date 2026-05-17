# Runbook: Promover un modelo manualmente

## Cuándo usarlo
- El DAG `validate_and_promote_model` falló pero el modelo pasó los quality gates manualmente.
- Se necesita hacer rollback a una versión anterior del modelo.
- Se está probando un modelo challenger y se quiere promover sin esperar el ciclo de reentrenamiento.

## Prerequisitos

```bash
source .env    # cargar POSTGRES_USER, POSTGRES_DB, etc.
```

## Paso 1 — Identificar la versión del modelo en MLflow

Abre `http://localhost:5000` → Model Registry → FraudDetectionModel.

O vía CLI:
```bash
curl -fsS "http://localhost:5000/api/2.0/mlflow/registered-models/get-latest-versions" \
  -H "Content-Type: application/json" \
  -d '{"name": "FraudDetectionModel", "stages": ["Staging", "Production"]}' \
  | python3 -m json.tool
```

Anota el `version` (número entero) y el `run_id` de la versión a promover.

## Paso 2 — Transicionar a Production en MLflow

```bash
curl -fsS -X POST "http://localhost:5000/api/2.0/mlflow/model-versions/transition-stage" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "FraudDetectionModel",
    "version": "<VERSION>",
    "stage": "Production",
    "archive_existing_versions": true
  }'
```

## Paso 3 — Registrar en PostgreSQL con el stored procedure

```bash
# Obtener el ID de la versión en model_deployments (si ya existe de un run previo):
docker compose exec -T postgresql psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c \
  "SELECT id, model_name, version, is_active FROM public.model_deployments ORDER BY id DESC LIMIT 5;"

# Si la versión no existe, insertarla primero:
docker compose exec -T postgresql psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c \
  "INSERT INTO public.model_deployments (model_name, version, mlflow_run_id, training_data_from, training_data_to)
   VALUES ('FraudDetectionModel', '<VERSION>', '<RUN_ID>', NOW() - INTERVAL '30 days', NOW())
   RETURNING id;"

# Activar con el stored procedure (reemplaza <ID> con el id obtenido):
docker compose exec -T postgresql psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c \
  "SELECT public.activate_model_version(<ID>);"
```

## Paso 4 — Recargar el modelo en FastAPI

```bash
# FastAPI carga el modelo Production al iniciar.
# Para que tome la nueva versión sin downtime completo:
docker compose up -d --force-recreate fastapi
# Verificar que cargó la versión correcta:
curl -fsS http://localhost:8000/model/info | python3 -m json.tool
```

## Rollback a versión anterior

Repite los pasos 1–4 con el `version` anterior. El stored procedure `activate_model_version` desactiva automáticamente todas las versiones previas.

## Verificación final

```bash
curl -fsS http://localhost:8000/health      # debe responder {"status": "ok"}
curl -fsS http://localhost:8000/model/info  # debe mostrar la nueva versión
```
