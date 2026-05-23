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
Consume del topic `transactions.raw`, computa features (ventana temporal + perfil histórico) y publica en `transactions.features`. Persiste en TimescaleDB y cachea estado en Redis para hidratación en caliente. Es un servicio long-running, se inicia con:

```bash
docker compose up -d consumer
```

### `inference/`
Consume del topic `transactions.features`, llama a la API FastAPI de inferencia, publica resultados en `transactions.predictions` y alertas de fraude en `fraud.alerts`. Incluye circuit breaker para proteger la API. Servicio long-running:

```bash
docker compose up -d inference
```

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
