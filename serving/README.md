# Serving

API de inferencia en tiempo real con FastAPI. Carga el modelo desde MLflow Registry, prepara features y expone endpoints de predicción single y batch.

## Arquitectura

```
request ──> FastAPI ──> ModelLoader (feature prep + XGBoost) ──> PredictionCache (Redis)
                │                                                     │
                └──> PredictionStore (asyncpg → PostgreSQL)           └──> cache hit → skip inference
```

## Endpoints

### `GET /health`

Estado del servicio.

```json
{ "status": "ok", "model_loaded": true }
```
`"degraded"` si el modelo no pudo cargarse (el servicio igual responde).

### `GET /model/info`

Metadata del modelo activo.

```json
{
  "model_name": "FraudDetectionModel",
  "model_version": "2",
  "model_stage": "Production",
  "loaded_at": "2026-05-16T10:00:00",
  "fraud_score_threshold": 0.5,
  "deployment_id": 1
}
```

### `POST /predict`

Predicción individual.

```json
{
  "transaction_id": "tx-001",
  "user_id": "user-42",
  "merchant_id": "merchant-7",
  "merchant_category": "electronics",
  "amount": 250.00,
  "country": "AR",
  "timestamp": "2026-05-16T10:00:00Z",
  "device_type": "mobile",
  "ip_hash": "a1b2c3d4",
  "features": {
    "tx_count_1h": 5.0,
    "tx_count_24h": 12.0,
    "tx_count_7d": 48.0,
    "amount_sum_1h": 500.00,
    "amount_sum_24h": 1200.50,
    "seconds_since_last_tx": 340.0,
    "amount_ratio_vs_user_avg": 1.1,
    "is_country_new": 0.0,
    "distinct_countries_seen": 2.0,
    "is_merchant_new": 0.0,
    "distinct_merchants_seen": 5.0
  }
}
```

El campo `features` contiene las 11 features de ventana y perfil histórico computadas upstream por `streaming.features`. Verifica cache en Redis por `transaction_id` antes de inferir.

**Response:**

```json
{
  "transaction_id": "tx-001",
  "prediction_score": 0.87,
  "prediction_label": true,
  "model_version": "2",
  "latency_ms": 12.3
}
```

### `POST /predict/batch`

Predicción batch (1 a 500 transacciones). Internamente hace `np.vstack` para inferencia vectorizada.

```json
{
  "predictions": [ { "..." }, { "..." } ],
  "total": 2,
  "latency_ms": 18.5
}
```

## Servicios

### `services/model_loader.py` — `ModelLoader`

Carga del modelo en el startup del lifecycle:

1. Conecta a MLflow Tracking, obtiene el último modelo en `Production`
2. Descarga artifacts a `/tmp/fraud_model/`
3. Carga XGBoost (`xgboost_model.joblib`) y el encoder categórico (`categorical_encoder.joblib`)
4. Consulta en PostgreSQL el `deployment_id` activo

**`prepare_features(raw, window_features)`** produce un array numpy de **17 features**:

| # | Feature | Fuente |
|---|---|---|
| 0 | `log1p(amount)` | `raw.amount` |
| 1 | `hour_of_day` | `raw.timestamp.hour` |
| 2 | `day_of_week` | `raw.timestamp.weekday()` |
| 3 | `merchant_category_encoded` | Target encoding con fallback a media global |
| 4 | `country_encoded` | Target encoding con fallback a media global |
| 5 | `device_type_encoded` | Ordinal encoding con fallback a -1 para valores no vistos |
| 6–16 | 11 features de ventana | `window_features` (recibidas de `streaming.features`) |

Si el modelo no está disponible, el servicio arranca en modo **degraded** (health check lo reporta, predict devuelve 503).

### `services/prediction_store.py` — `PredictionStore`

Persiste asincrónicamente cada predicción en `public.predictions_history` via `asyncpg`. No bloquea la response (fire-and-forget con `background_tasks`).

### `services/prediction_cache.py` — `PredictionCache`

Cache en Redis con TTL de 60s. Si Redis no está disponible, degrada gracefulmente (desactiva cache, no falla).

## Levantar

```bash
docker compose up -d serving
```

La API queda en `http://localhost:8000` con docs interactivos en `/docs` y `/redoc`.
