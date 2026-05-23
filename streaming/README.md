# Streaming

Simula y procesa el flujo de transacciones del sistema de detección de fraude. Es el punto de entrada de datos: genera transacciones raw, las transforma a features, consulta el modelo y publica predicciones.

## Arquitectura

```
producer (Kafka) ──raw──> features ──features──> inference ──predictions──> (alerts, serving)
                               │                      │
                               └── TimescaleDB        └── FastAPI (model)
```

Tres procesos independientes que se comunican via Kafka con esquemas Avro:

### `producer/`
Publica transacciones simuladas en el topic `transactions.raw`. Soporta dos modos:

- **`live`** — flujo continuo con tasa de fraude configurable (`--fraud-rate`). Mezcla transacciones legítimas con patrones fraudulentos aleatorios.
- **`scenario`** — inyecta un patrón de fraude específico (`amount_anomaly`, `unusual_country`, `high_frequency`, `unknown_merchant`, `mixed`). Usa una tasa de fraude fija del 25%.

Estructura interna:

| Archivo | Responsabilidad |
|---|---|
| `generator.py` | Genera datos sintéticos: transacciones legítimas (`LegitimateTransactionGenerator`) y patrones de fraude (`FraudPatternGenerator`). No depende de Kafka. |
| `transaction_producer.py` | Clase `TransactionProducer` que serializa y envía una `Transaction` a Kafka usando Avro. No sabe cómo se generan los datos. |
| `main.py` | Orquesta: parsea args CLI, conecta el generator con el producer Kafka, aplica rate limiting y maneja señales. Punto de entrada. |

Esta separación sigue el principio de responsabilidad única: cada módulo tiene una sola razón de cambio.

Todos los servicios corren dentro de contenedores Docker. El producer requiere override del CMD para pasar argumentos:

```bash
# Modo live: mezcla legítimas + fraude
docker compose run --rm producer python -m streaming.producer.main --mode live --tps 10 --fraud-rate 0.02

# Modo scenario: patrón específico
docker compose run --rm producer python -m streaming.producer.main --mode scenario --scenario high_frequency --tps 5
```

| Flag | Default | Descripción |
|---|---|---|
| `--mode` | `live` | Modo de operación: `live` (flujo continuo) o `scenario` (patrón fijo) |
| `--tps` | `10` | Transacciones por segundo objetivo |
| `--duration` | `0` | Duración en segundos (`0` = infinito, hasta Ctrl+C) |
| `--fraud-rate` | `0.02` | Proporción de fraude en modo `live` (0.0 a 1.0) |
| `--scenario` | — | Patrón de fraude para modo `scenario`: `amount_anomaly`, `unusual_country`, `high_frequency`, `unknown_merchant`, `mixed` |
| `--seed` | `42` | Seed para reproducibilidad de datos generados |
| `--num-users` | `200` | Cantidad de usuarios simulados |
| `--num-merchants` | `50` | Cantidad de merchants disponibles |

### `features/`
Consume del topic `transactions.raw`, computa features (ventana temporal + perfil histórico) y publica en `transactions.features`. Persiste transacciones en TimescaleDB y cachea estado por usuario en Redis para sobrevivir reinicios. Es un servicio long-running:

```bash
docker compose up -d zookeeper kafka redis timescaledb
docker compose up -d features
```

Estructura interna:

| Archivo | Responsabilidad |
|---|---|
| `feature_types.py` | Dataclasses inmutables `WindowFeatures` e `HistoricalFeatures`. Definen el contrato de salida del cómputo; no tienen lógica. |
| `sliding_window_store.py` | `SlidingWindowStore` — mantiene en memoria un deque de transacciones por usuario y computa conteos/sumas en ventanas de 1h, 24h y 7d. Evicta automáticamente transacciones fuera de la ventana máxima. |
| `historical_profile_store.py` | `HistoricalProfileStore` — mantiene en memoria el perfil acumulado del usuario (promedio de monto, países y merchants vistos) y computa features de anomalía como `amount_ratio_vs_user_avg` e `is_country_new`. |
| `transaction_consumer.py` | `TransactionConsumer` — consumer Kafka con deserialización Avro, commit manual por mensaje y retry queue de un intento antes de dead-letter. |
| `feature_publisher.py` | `FeaturePublisher` — extiende `AvroPublisher` y serializa la transacción original junto con las features computadas en el schema `transaction_features.avsc`. |
| `user_store.py` | `UserStore` (Redis) — persiste y recupera el estado en memoria de cada usuario. Se lee **una sola vez por usuario** al arrancar (hidratación en caliente); se escribe en cada transacción. Degradación elegante: si Redis no está disponible el servicio continúa sin persistencia de estado. |
| `transaction_store.py` | `TransactionStore` (TimescaleDB) — inserta cada transacción en la hypertable `public.transactions` con `ON CONFLICT DO NOTHING`. Solo escribe; nunca lee. Los campos `is_fraud`, `model_score` y `latency_ms` quedan en `NULL` hasta que el servicio `inference` los complete. Degradación elegante si TimescaleDB no está disponible. |
| `main.py` | Orquesta el loop principal: consume, hidrata estado si es usuario nuevo, computa features, persiste en Redis y TimescaleDB, publica en Kafka. Maneja señales `SIGINT`/`SIGTERM` para shutdown limpio. |

### `inference/`
Consume del topic `transactions.features`, llama a la API de serving para obtener la predicción, publica resultados en `transactions.predictions` y alertas en `transactions.fraud.alerts`. Protege la API con un circuit breaker. Servicio long-running:

```bash
docker compose up -d inference
```

Estructura interna:

| Archivo | Responsabilidad |
|---|---|
| `feature_consumer.py` | `FeatureConsumer` — consumer Kafka con deserialización Avro, commit manual y retry queue de un intento antes de dead-letter. Análogo a `TransactionConsumer` pero lee del topic `transactions.features`. |
| `api_client.py` | `InferenceApiClient` — cliente HTTP (`httpx`) que llama a `GET /model/info` al arrancar para obtener el `deployment_id` activo, y a `POST /predict` por cada mensaje consumido. Timeout de 2 s por defecto. |
| `circuit_breaker.py` | `CircuitBreaker` — implementa el patrón CLOSED → OPEN → HALF_OPEN. Se abre tras N fallos consecutivos contra la API y espera un cooldown configurable antes de intentar de nuevo. |
| `prediction_publisher.py` | `PredictionPublisher` — extiende `AvroPublisher` y serializa el resultado de la predicción (`score`, `label`, `model_version_id`, `latency_ms`) en el schema `transaction_prediction.avsc`. |
| `alert_publisher.py` | `AlertPublisher` — extiende `AvroPublisher` y publica una alerta de fraude en `transactions.fraud.alerts` cuando `prediction_label` es `True`. Clasifica la severidad en `WARNING` (score < 0.75), `HIGH` (≥ 0.75) o `CRITICAL` (≥ 0.90). |
| `main.py` | Orquesta el loop principal: consume un mensaje, verifica el circuit breaker, llama a la API, publica predicción, publica alerta si hay fraude, hace commit del offset. Maneja señales `SIGINT`/`SIGTERM` para shutdown limpio. |

## Schemas (Avro)

| Archivo | Topic | Descripción |
|---|---|---|
| `transaction_raw.avsc` | `transactions.raw` | Transacción cruda del producer |
| `transaction_features.avsc` | `transactions.features` | Features computadas por el servicio features |
| `transaction_prediction.avsc` | `transactions.predictions` | Resultado de inferencia |
| `fraud_alert.avsc` | `fraud.alerts` | Alerta cuando se detecta fraude |

## Shared

- **`AvroPublisher`** (`publisher.py`) — base class reutilizable para publicadores Avro con compresión snappy, idempotencia y delivery callbacks.
- **`Transaction`** (`models.py`) — dataclass inmutable compartida entre producer y features.
