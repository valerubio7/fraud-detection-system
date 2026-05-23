# Tests

## Cómo ejecutar

```bash
# Unitarios (~7s)
.venv/bin/pytest tests/unit/

# Integración sin Docker (~3s)
.venv/bin/pytest tests/integration/ -m "not integration"

# Integración con Docker (~55s)
.venv/bin/pytest tests/integration/ -m "integration"

# Carga con Docker (~70s)
.venv/bin/pytest tests/load/

# Todo junto
.venv/bin/pytest tests/unit/ tests/integration/ tests/load/
```

---

## Unit

Tests rápidos sin dependencias externas. PostgreSQL, Redis, Kafka, MLflow y pandas se reemplazan con stubs, por lo que no requieren Docker ni servicios corriendo.

Cubren toda la lógica de negocio del sistema: feature engineering (ventanas temporales, perfiles históricos, selección y encoding), el pipeline de entrenamiento offline (métricas, balanceo de clases), el servidor de inferencia (endpoint `/predict`, cache, circuit breaker) y los DAGs de Airflow (estructura, dependencias y lógica interna de las tasks).

---

## Integration

Verifican que los distintos componentes funcionan correctamente cuando se conectan a servicios reales. Los tests sin `@pytest.mark.integration` solo necesitan las librerías instaladas (Evidently, scikit-learn); el resto levanta contenedores Docker con testcontainers.

Cubren: persistencia en TimescaleDB y PostgreSQL, caché en Redis, producción y consumo de mensajes Avro en Kafka, el pipeline completo de entrenamiento con MLflow, detección de drift con Evidently y el endpoint `/predict` con base de datos y caché reales.

---

## Load

Tests de rendimiento que verifican que el sistema cumple sus SLAs bajo carga. Todos requieren Docker.

- **`test_timescaledb_benchmarks.py`** — benchmarks de las queries críticas contra 100.000 filas reales. Verifica que índices y continuous aggregates mantienen los tiempos de respuesta por debajo del umbral (< 50ms para queries indexadas).
- **`test_kafka_throughput.py`** — mide el throughput del pipeline Kafka. El objetivo es procesar al menos 10.000 transacciones/minuto a través del pipeline completo (consumer → feature engineering → publisher).
- **`locustfile.py`** — test de carga HTTP con Locust. Simula hasta 500 usuarios concurrentes contra `/predict` y valida P99 < 100ms y error rate < 1%. Requiere el stack completo corriendo con `docker compose up`.
