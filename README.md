# Fraud Detection System — MLOps End-to-End

Sistema de detección de fraude en tiempo real con plataforma MLOps completa: streaming con Kafka, feature engineering online, inferencia XGBoost < 100ms, reentrenamiento automático con Airflow, drift detection con Evidently AI, y observabilidad con Grafana + Prometheus.

> Replica la arquitectura de sistemas como Mercado Pago o Visa: cada transacción se evalúa en menos de 100ms mientras el modelo se monitorea solo y se reentrena cuando empieza a degradarse.

## Arquitectura en una línea

```
Simulador → [Kafka: transactions.raw] → Feature Engineering → [Kafka: transactions.features]
  → FastAPI / XGBoost → [Kafka: transactions.predictions] → Grafana Alertas
                   ↓
             TimescaleDB ← transacciones + features
             PostgreSQL  ← predicciones + métricas + drift
                   ↑
         Airflow (reentrenamiento diario) + Evidently (drift cada 6h) + MLflow (registry)
```

→ Diagrama detallado: [docs/architecture.md](docs/architecture.md)

## Stack tecnológico

| Capa | Tecnología | Rol |
|---|---|---|
| Streaming | Apache Kafka | Pipeline de transacciones en tiempo real |
| Feature Engineering | Python (custom) | Ventanas deslizantes y features históricas online |
| Modelo | XGBoost | Clasificador binario de fraude |
| Serving | FastAPI + uvicorn | Inferencia con latencia P99 < 100ms |
| ML Tracking | MLflow | Experimentos, versiones y model registry |
| Orquestación | Apache Airflow | Reentrenamiento automático y validación |
| Drift Detection | Evidently AI | Detección de data drift y model drift |
| DB Series Temporales | TimescaleDB | Transacciones con hypertables y continuous aggregates |
| DB Relacional | PostgreSQL | Predicciones, métricas, audit log, stored procedures |
| Monitoreo | Grafana + Prometheus | 4 dashboards y 4 alertas en tiempo real |
| Infra | Docker + Compose | Stack completo en un solo comando |
| CI/CD | GitHub Actions | Lint, tests, build y security scan |

## Quickstart

**Prerequisitos:** Docker Desktop (o Docker Engine + Compose v2) y `curl`.

```bash
# 1. Clonar el repo
git clone https://github.com/valerubio7/fraud-detection-system.git
cd fraud-detection-system

# 2. Primer setup — crea .env, construye imágenes, levanta el stack completo
./scripts/setup.sh
# Si es la primera vez, el script crea .env desde .env.example y pide
# que edites las credenciales. Luego volvé a correr el mismo comando.

# 3. Verificar que todo está healthy
./scripts/smoke_test.sh

# 4. Iniciar el simulador de transacciones
uv run python -m streaming.producer.main --mode live --tps 10 --fraud-rate 0.02

# 5. Abrir los dashboards
open http://localhost:3000   # Grafana (admin/admin por defecto)
```

## Servicios disponibles

| Servicio | URL | Descripción |
|---|---|---|
| **FastAPI** | http://localhost:8000/docs | API de inferencia (Swagger UI) |
| **FastAPI ReDoc** | http://localhost:8000/redoc | Documentación alternativa |
| **MLflow** | http://localhost:5000 | Experimentos y model registry |
| **Airflow** | http://localhost:8081 | DAGs de reentrenamiento y drift |
| **Grafana** | http://localhost:3000 | Dashboards de fraude y sistema |
| **Prometheus** | http://localhost:9090 | Métricas de FastAPI |
| **Kafka UI** | http://localhost:8080 | Topics y consumer groups |
| **PostgreSQL** | localhost:5432 | Metadata del sistema |
| **TimescaleDB** | localhost:5433 | Series temporales de transacciones |

## Flujo de datos detallado

### Pipeline en tiempo real (< 100ms por transacción)

1. **Producer** genera transacciones sintéticas (modos: `live`, `scenario`, `replay`) y publica en `transactions.raw`.
2. **Features** (feature engineering online): consume `transactions.raw`, calcula features en ventanas deslizantes (1h/24h/7d) usando `SlidingWindowStore` y `HistoricalProfileStore`, publica en `transactions.features`, y escribe en TimescaleDB.
3. **Inference**: consume `transactions.features`, llama a FastAPI `/predict`, publica el resultado en `transactions.predictions` y alertas en `transactions.fraud.alerts`.
4. **FastAPI**: recibe `TransactionRequest`, aplica el feature pipeline, infiere con XGBoost, y guarda la predicción en PostgreSQL de forma asíncrona.

### Pipeline MLOps (batch / scheduled)

| DAG | Schedule | Qué hace |
|---|---|---|
| `retrain_fraud_model` | Diario 2 AM | Extrae datos, entrena XGBoost, registra en MLflow |
| `validate_and_promote_model` | Triggered | Quality gates → promote a Production o archive |
| `drift_detection_report` | Cada 6h | Evidently AI → guarda en PostgreSQL → trigger si drift > 0.3 |
| `data_quality_check` | Cada 1h | Verifica volumen del stream y distribución de amounts |

### Quality gates del modelo

| Métrica | Umbral mínimo |
|---|---|
| F1-score | >= 0.85 |
| AUC-ROC | >= 0.90 |
| Latencia P99 (batch 1000) | <= 50ms |
| Mejora sobre champion | > 2% en F1 |

## Estructura del proyecto

```
fraud-detection-system/
├── streaming/
│   ├── producer/          # Simulador de transacciones (Kafka producer)
│   ├── features/          # Feature engineering online + writer TimescaleDB
│   ├── inference/         # Consumer features → FastAPI → predictions
│   └── schemas/           # Schemas Avro para los topics Kafka
├── offline_features/      # Pipeline batch para entrenamiento (featurizer, encoders)
├── model/
│   ├── train.py           # Pipeline de entrenamiento XGBoost
│   ├── evaluate.py        # Quality gates y métricas
│   └── selected_features.py # Lista canónica de 16 features
├── serving/
│   └── app/               # FastAPI: routes, schemas, services, model_loader
├── mlops/
│   ├── airflow/dags/      # 4 DAGs de Airflow
│   └── evidently/         # Drift detection: data_drift, model_drift, drift_policy
├── database/
│   ├── timescaledb/       # Migraciones + seeds + hypertable schema
│   └── postgresql/        # Migraciones + stored procedures + triggers
├── monitoring/
│   ├── grafana/           # 4 dashboards JSON + provisioning YAML
│   └── prometheus/        # prometheus.yml
├── docker/                # Dockerfiles por servicio
├── scripts/
│   ├── setup.sh           # Setup inicial (dev local, con build)
│   ├── deploy.sh          # Bootstrap desde imágenes pre-construidas
│   ├── smoke_test.sh      # Verificación post-deploy
│   └── lib.sh             # Funciones compartidas entre scripts
├── tests/
│   ├── unit/              # Tests unitarios (pytest, sin servicios externos)
│   ├── integration/       # Tests de integración (testcontainers)
│   └── load/              # Benchmarks Locust y TimescaleDB
├── docs/
│   ├── architecture.md    # Diagrama y decisiones de diseño
│   ├── glossary.md        # Glosario técnico del proyecto
│   ├── implementation-story.md # Historia de implementación fase a fase
│   └── runbooks/          # Procedimientos operativos
└── .github/workflows/     # CI/CD: lint, tests, build, security scan
```

## Desarrollo local

```bash
# Instalar dependencias de desarrollo
uv sync --group serving --group consumer --group model --group testing

# Activar hooks de pre-commit (ruff + check-yaml + detect-private-key)
uv run pre-commit install

# Correr tests unitarios
uv run pytest tests/unit/ -v

# Correr linter
uvx ruff check .
uvx ruff format --check .
```

Ver [CONTRIBUTING.md](CONTRIBUTING.md) para convenciones de código y proceso de PR.

## Documentación

- [Arquitectura detallada](docs/architecture.md) — diagrama Mermaid, decisiones de diseño, alternativas consideradas
- [Glosario técnico](docs/glossary.md) — definiciones de todos los conceptos del sistema
- [Historia de implementación](docs/implementation-story.md) — decisiones técnicas fase a fase
- [Runbooks](docs/runbooks/) — procedimientos operativos (restart, promote model, kafka lag, backup)
- [API Reference](http://localhost:8000/redoc) — documentación OpenAPI (requiere stack corriendo)

## Licencia

MIT
