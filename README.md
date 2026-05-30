# Sistema de Detección de Fraude en Tiempo Real

Sistema de detección de fraude bancario que procesa transacciones en tiempo real mediante un pipeline de streaming sobre Apache Kafka. Cada transacción pasa por una etapa de feature engineering que calcula ventanas deslizantes (1h, 24h, 7d) y perfiles históricos por usuario almacenados en Redis, para luego ser evaluada por un modelo XGBoost servido con FastAPI. Las predicciones y alertas se publican de vuelta a Kafka y se persisten en PostgreSQL.

El sistema incluye un pipeline MLOps completo orquestado por Apache Airflow: reentrenamiento diario del modelo con datos de TimescaleDB, evaluación con quality gates (F1 ≥ 0.85, AUC-ROC ≥ 0.90), promoción automática al model registry de MLflow y detección de drift cada 6 horas con Evidently AI. Cuando el drift supera el umbral configurado, Airflow dispara automáticamente un nuevo ciclo de reentrenamiento sin intervención manual.

Todo el stack —19 servicios— corre sobre Docker Compose y se levanta con un único script que incluye seed de datos, entrenamiento inicial del modelo y verificación de salud de todos los servicios.

---

## Arquitectura

### Pipeline en tiempo real

```
Simulador        Kafka               Feature              Kafka             Inference
Producer    →  transactions.raw  →  Engineering      →  transactions   →  Consumer
                                    Consumer             .features         │
                                    │                                      │ POST /predict
                                    ├─ Redis (estado ventanas)             │
                                    └─ TimescaleDB (persistencia)          FastAPI + XGBoost
                                                                           │
                                                              transactions.predictions
                                                              transactions.fraud.alerts
```

### Pipeline MLOps (batch — orquestado por Airflow)

```
TimescaleDB ──► retrain_fraud_model (diario 2 AM) ──► MLflow Registry
                                                            │
                                               validate_and_promote_model
                                                            │ (quality gates: F1 ≥ 0.85, AUC-ROC ≥ 0.90)
                                                    PostgreSQL model_deployments
                                                            │
                                                     FastAPI carga modelo nuevo

TimescaleDB ──► drift_detection_report (cada 6h) ──► Evidently AI
                                                            │ (drift > 0.30)
                                                     dispara reentrenamiento
```

### Monitoreo

```
FastAPI ──► Prometheus ──► Grafana ◄── TimescaleDB
                                  ◄── PostgreSQL
                    Alertas unificadas (Grafana Unified Alerting)
```

---

## Stack tecnológico

| Tecnología | Versión | Rol |
|-----------|---------|-----|
| Python | 3.11+ | Lenguaje principal |
| FastAPI + Uvicorn | 0.136 / 0.44 | API REST de inferencia |
| XGBoost | 2.1.4 | Modelo de clasificación de fraude |
| PostgreSQL | 16.2 | Metadata del sistema: despliegues, predicciones, alertas, drift |
| TimescaleDB | 2.14.2-pg16 | Serie temporal de transacciones (hypertable) |
| Redis | 7.2 | Caché de features en streaming y predicciones |
| Apache Kafka | 7.6.0 (Confluent) | Bus de eventos entre productores y consumidores |
| Apache Airflow | 2.11.0 | Orquestación de pipelines de reentrenamiento y drift |
| MLflow | 2.17.2 | Tracking de experimentos y model registry |
| Evidently AI | 0.4.36 | Detección de drift de datos y modelo |
| Prometheus | 2.51.0 | Recolección de métricas |
| Grafana | latest | Dashboards y alertas |
| Docker + Compose v2 | — | Contenedorización de todos los servicios |
| asyncpg / psycopg2 | 0.31 / 2.9 | Acceso directo a PostgreSQL sin ORM |

---

## Guía de instalación

### Prerrequisitos

- **Docker** (versión reciente con soporte Compose v2)
- **Docker Compose v2** — se verifica con `docker compose version` (debe imprimir `v2.x.x` o superior)
- **curl** y **Python 3** disponibles en el host (usados por el script de setup)
- Al menos **8 GB de RAM** disponibles para el stack completo
- Al menos **10 GB de espacio en disco** para imágenes y datos

---

### Paso 1 — Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd fraud-detection-system
```

---

### Paso 2 — Configurar las variables de entorno

```bash
cp .env.example .env
```

El archivo ya tiene valores por defecto listos para desarrollo local. No es necesario modificar nada para levantar el stack.

> Para un entorno que no sea de desarrollo, se recomienda reemplazar las contraseñas del `.env` antes de continuar.

---

### Paso 3 — Ejecutar el script de setup

```bash
./scripts/setup.sh
```

El script es **idempotente** — se puede correr más de una vez sin duplicar datos ni reentrenar si el modelo ya existe. Ejecuta 6 etapas en secuencia:

| Etapa | Descripción | Tiempo estimado |
|-------|-------------|-----------------|
| 1/6 | Verificación de prerrequisitos y variables de entorno | < 10 s |
| 2/6 | Build de imágenes Docker locales | 5–15 min (primera vez) |
| 3/6 | Arranque de infraestructura base e inicialización de MLflow | 1–2 min |
| 4/6 | Inicialización de Airflow, topics Kafka y migraciones SQL | 2–3 min |
| 5/6 | Seed de 100.000 transacciones, entrenamiento XGBoost y promoción del modelo | 5–10 min |
| 6/6 | Verificación final de todos los servicios | < 1 min |

**Tiempo total primera ejecución: ~5–20 minutos** (varía según la velocidad de la conexión y el hardware).

Al finalizar, el script imprime las URLs de todos los servicios y el modelo en producción.

---

### Paso 4 — Verificar que todo está funcionando

```bash
# Estado de todos los contenedores
docker compose ps
```

---

## Servicios disponibles

| Servicio | URL | Descripción |
|---------|-----|-------------|
| **API de inferencia** | http://localhost:8000 | Endpoint principal de predicción de fraude |
| **Swagger UI** | http://localhost:8000/docs | Documentación interactiva de la API |
| **MLflow** | http://localhost:5000 | Experimentos, métricas y model registry |
| **Airflow** | http://localhost:8081 | DAGs de reentrenamiento y drift (usuario: `admin`) |
| **Prometheus** | http://localhost:9090 | Métricas del sistema |
| **Grafana** | http://localhost:3000 | Dashboards de monitoreo (usuario: `admin`) |
| **Kafka UI** | http://localhost:8080 | Inspección de topics (solo en desarrollo) |

---

## Uso de la API

### Predicción individual

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
    "user_id": "user_AR_001",
    "merchant_id": "merchant_supermaxi",
    "merchant_category": "grocery",
    "amount": 150.75,
    "country": "AR",
    "timestamp": "2025-01-15T14:30:00Z",
    "device_type": "mobile",
    "ip_hash": "a3f8b2c1d4e5",
    "features": {
      "tx_count_1h": 3.0,
      "tx_count_24h": 10.0,
      "tx_count_7d": 52.0,
      "amount_sum_1h": 320.50,
      "amount_sum_24h": 1080.00,
      "seconds_since_last_tx": 1800.0,
      "amount_ratio_vs_user_avg": 1.4,
      "is_country_new": 0.0,
      "is_merchant_new": 0.0,
      "distinct_merchants_seen": 8.0
    }
  }'
```

> El campo `ip_hash` es obligatorio. Las `features` son pre-computadas por el pipeline de streaming; en un entorno con el stack corriendo, el `inference` consumer las envía automáticamente.

Respuesta:

```json
{
  "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
  "prediction_score": 0.476,
  "prediction_label": false,
  "model_version": "1",
  "latency_ms": 12.49
}
```

---

## Streaming en tiempo real

El script `./scripts/streaming.sh` controla el pipeline de eventos interactivamente:

```bash
./scripts/streaming.sh
```

---

## Estructura del repositorio

```
fraud-detection-system/
├── database/
│   ├── postgresql/
│   │   ├── migrations/          # Esquema: model_deployments, predictions_history, drift_reports, alert_log
│   │   ├── stored_procedures/   # activate_model_version (con FOR UPDATE), check_fraud_rate
│   │   └── triggers/            # alert_on_high_fraud_rate (AFTER INSERT)
│   └── timescaledb/
│       ├── migrations/          # Hypertable transactions, vistas materializadas continuas, compresión
│       └── seeds/               # Generador de 100.000 transacciones sintéticas
├── docker/                      # Dockerfiles por servicio
├── docs/
│   └── architecture.md          # Diagramas Mermaid de la arquitectura
├── mlops/
│   ├── airflow/dags/            # retrain, validate_and_promote, drift_detection, data_quality
│   ├── evidently/               # Lógica de detección de drift de datos y modelo
│   └── mlflow/                  # Inicialización del experimento y registry
├── model/
│   └── pipeline/                # train.py, evaluate.py, promote.py
├── monitoring/
│   ├── grafana/                 # Dashboards: system_health, drift_monitor
│   └── prometheus/              # prometheus.yml con scrape configs
├── offline_features/            # Feature engineering offline: encoders, featurizer, selección
├── scripts/
│   ├── setup.sh                 # Setup completo del entorno (idempotente, 6 etapas)
│   ├── deploy.sh                # Despliegue con imágenes pre-built del registry
│   └── streaming.sh             # Control interactivo del pipeline de streaming
├── serving/
│   └── app/                     # FastAPI: rutas /predict, /predict/batch, /health
├── streaming/
│   ├── producer/                # Simulador de transacciones → Kafka
│   ├── features/                # Consumer: computa features → TimescaleDB + Redis
│   └── inference/               # Consumer: llama a la API → publica predicciones y alertas
├── tests/
│   ├── unit/                    # Tests unitarios sin dependencias externas
│   ├── integration/             # Tests con contenedores reales (testcontainers)
│   └── load/                    # Benchmarks de throughput (Locust, TimescaleDB)
├── .env.example                 # Plantilla de variables de entorno
├── docker-compose.yml           # Definición completa del stack
└── pyproject.toml               # Dependencias y configuración de herramientas
```

---

## Comandos útiles

```bash
# Ver logs de un servicio en tiempo real
docker compose logs -f serving
docker compose logs -f airflow-scheduler

# Detener el stack (preserva datos en volúmenes)
docker compose down

# Detener el stack y eliminar todos los datos
docker compose down -v

# Conectarse a PostgreSQL
docker compose exec postgresql psql -U fraud_metadata_user -d fraud_metadata

# Conectarse a TimescaleDB
docker compose exec timescaledb psql -U fraud_timeseries_user -d fraud_transactions_timeseries

# Ver el modelo activo en producción
docker compose exec postgresql psql -U fraud_metadata_user -d fraud_metadata \
  -c "SELECT id, model_name, version, f1_score, created_at FROM model_deployments WHERE is_active;"
```

---

## Documentación por módulo

| Módulo | README | Descripción |
|--------|--------|-------------|
| `database/` | [database/README.md](database/README.md) | Esquemas, migraciones, funciones, triggers e índices de PostgreSQL y TimescaleDB |
| `streaming/` | [streaming/README.md](streaming/README.md) | Producer de transacciones, feature engineering consumer e inference consumer sobre Kafka |
| `serving/` | [serving/README.md](serving/README.md) | API FastAPI: endpoints `/predict`, `/predict/batch` y `/health`, caché Redis, persistencia async |
| `model/` | [model/README.md](model/README.md) | Pipeline de entrenamiento, evaluación con quality gates y promoción del modelo a MLflow |
| `offline_features/` | [offline_features/README.md](offline_features/README.md) | Feature engineering offline: encoders, selección de features y estrategias de imbalance |
| `mlops/` | [mlops/README.md](mlops/README.md) | DAGs de Airflow, detección de drift con Evidently AI e inicialización de MLflow |
| `tests/` | [tests/README.md](tests/README.md) | Tests unitarios, de integración con contenedores reales y benchmarks de carga |

---

## Contexto académico

**Materia:** Bases de Datos Avanzada
**Trabajo:** Trabajo Final Grupal

El proyecto implementa **6 de los 7 temas** requeridos por la cátedra:

| # | Tema | Implementación | Justificación |
|---|------|---------------|---------------|
| 1 | **Índices** | 12 índices custom en PostgreSQL y TimescaleDB: compuestos (`user_id, timestamp`), parciales (`WHERE is_active IS TRUE`, `WHERE acknowledged_at IS NULL`, `WHERE is_fraud IS TRUE`) | Las consultas críticas filtran por estado activo del modelo y ventanas temporales recientes. Los índices parciales reducen el tamaño del índice y aceleran los filtros más frecuentes evitando full scans sobre tablas grandes. |
| 2 | **Particionado** | TimescaleDB hypertable sobre `transactions` con chunks diarios (`INTERVAL '1 day'`), política de compresión automática a los 7 días y retención de 2 años | Las transacciones crecen ilimitadamente en el tiempo. El particionado temporal permite que las queries sobre ventanas recientes toquen solo los chunks relevantes; la compresión reduce el almacenamiento de datos históricos hasta un 90%. |
| 3 | **Transacciones** | Stored procedure `activate_model_version` con `SELECT ... FOR UPDATE` (locking pesimista) + trigger `check_fraud_rate` que corre dentro de una transacción AFTER INSERT | La activación de un nuevo modelo debe ser atómica: no puede quedar un instante sin modelo activo ni dos activos a la vez. El `FOR UPDATE` previene race conditions ante activaciones concurrentes. |
| 4 | **Seguridad** | Todas las credenciales se gestionan exclusivamente con variables de entorno (`.env`); la API valida todos los inputs con Pydantic antes de procesarlos; las queries usan parámetros en lugar de concatenación de strings | Nunca se hardcodean passwords en el código ni en las imágenes Docker. Los schemas Pydantic rechazan inputs malformados en la capa HTTP antes de que lleguen a la base de datos. |
| 5 | **Sin ORM** | Acceso directo a PostgreSQL y TimescaleDB con `asyncpg` (async) y `psycopg2` en toda la capa de datos de la aplicación | El modelo de datos es simple y estable; un ORM agregaría overhead sin beneficio. `asyncpg` permite queries concurrentes sin bloquear el event loop de FastAPI, crítico para la latencia de inferencia. |
| 6 | **NoSQL** | Redis como store de estado para las features en streaming (ventanas deslizantes de 1h/24h/7d y perfiles históricos por usuario) y como caché de predicciones por `transaction_id` | Redis permite leer y actualizar el estado de un usuario en < 1ms desde múltiples workers del consumidor de Kafka, lo que sería imposible consultando PostgreSQL por cada evento. Se combina con las bases relacionales: Redis para estado efímero caliente, PostgreSQL/TimescaleDB para datos persistentes. |
