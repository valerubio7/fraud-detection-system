# Arquitectura del Sistema

## Pipeline en tiempo real

```mermaid
flowchart LR
    SIM[Simulador\nProducer] -->|Avro| RAW[transactions.raw]
    RAW --> FE[Feature Engineering\nConsumer]
    FE -->|ventanas 1h/24h/7d| REDIS[(Redis\ncaché estado)]
    FE -->|inserta| TSDB[(TimescaleDB\nhypertable)]
    FE -->|Avro| FEAT[transactions.features]
    FEAT --> IC[Inference\nConsumer]
    IC -->|POST /predict| API[FastAPI\nXGBoost]
    API -->|async| PG[(PostgreSQL\npredicciones)]
    IC -->|Avro| PRED[transactions.predictions]
    IC -->|si fraude| ALERTS[transactions.fraud.alerts]
    API --> METRICS[/metrics\nPrometheus]
```

## Pipeline MLOps (batch)

```mermaid
flowchart TD
    TSDB[(TimescaleDB)] -->|extract| TRAIN[retrain_fraud_model DAG\ndiario 2 AM]
    TRAIN -->|registra| MLFLOW[(MLflow\nRegistry)]
    MLFLOW --> PROMOTE[validate_and_promote_model DAG]
    PROMOTE -->|quality gates OK| PG[(PostgreSQL\nmodel_deployments)]
    PROMOTE -->|activa| API[FastAPI\ncarga nuevo modelo]

    TSDB -->|últimas 24h| DRIFT[drift_detection_report DAG\ncada 6h]
    PG -->|referencia| DRIFT
    DRIFT -->|Evidently AI| DREP[drift_reports\nPostgreSQL]
    DREP -->|si drift > 0.3| TRAIN
```

## Monitoreo

```mermaid
flowchart LR
    API -->|scrape 10s| PROM[Prometheus]
    PROM --> GRAFANA[Grafana]
    TSDB[(TimescaleDB)] --> GRAFANA
    PG[(PostgreSQL)] --> GRAFANA
    GRAFANA -->|alertas| ALERT[Grafana\nUnified Alerting]
```
