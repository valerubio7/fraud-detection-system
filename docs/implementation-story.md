# Historia de implementacion

Este documento narra, de forma detallada, todo lo que se fue construyendo desde el inicio del proyecto hasta la finalizacion de la Fase 4 (modelo ML con XGBoost). El foco esta en decisiones, artefactos creados y el orden en que se consolidaron.

## Momento cero: vision y plan base

El proyecto arranco con una vision clara: deteccion de fraude en tiempo real con latencia baja y una plataforma MLOps completa (tracking, reentrenamiento, drift y monitoreo). Para asegurar ejecucion ordenada se definio un plan maestro en `docs/PLAN.md` con fases, tareas y tiempos estimados. Ese plan fijo el stack tecnico, las responsabilidades por capa y el flujo principal de datos.

Desde este punto se establecio un criterio: construir primero una base operativa solida (infraestructura, bases de datos y streaming) antes de atacar el entrenamiento del modelo y el serving avanzado. Esto reduce riesgos y habilita pruebas end-to-end tempranas.

## Fase 1: setup e infraestructura base

### 1.1 Estructura del repositorio y configuracion base

Se organizo el repositorio con carpetas separadas por dominio (ingestion, database, docker, serving, mlops, monitoring). La estructura quedo alineada con la arquitectura definida en el plan.

Se estandarizo la configuracion central en `config.py` usando Pydantic Settings. Esta decision permite validar variables criticas y compartir la misma fuente de verdad entre servicios. Se definieron settings para Kafka, PostgreSQL, TimescaleDB, Redis, MLflow, modelo, Airflow y Grafana.

### 1.2 Dockerizacion del stack

Se construyo un `docker-compose.yml` completo que levanta:

- Kafka + Zookeeper
- Schema Registry
- Kafka UI
- Redis
- TimescaleDB
- PostgreSQL
- MLflow
- Airflow (webserver y scheduler)
- FastAPI
- Grafana

Cada servicio cuenta con healthchecks para garantizar que el bootstrap sea confiable. Se definio una red interna `mlops-net` y volumenes persistentes para bases de datos, MLflow y Grafana.

Se agregaron Dockerfiles especificos para:

- FastAPI (Python 3.11, uv, user no-root, healthcheck)
- Airflow (imagen oficial, dependencias MLflow/Evidently/psycopg2)
- Kafka producer (Python 3.11 con dependencias de ingestion)
- Grafana (provisioning de datasources y dashboards)
- MLflow (dependencia psycopg2 para backend store)

Adicionalmente se incluyo `docker-compose.override.yml` para desarrollo local con montajes de codigo, logs mas verbosos y un listener extra de Kafka para debugging.

### 1.3 Script de setup idempotente

Se creo `scripts/setup.sh` como entrada unica para el bootstrap. Este script:

- Valida prerequisitos (Docker + Compose)
- Genera `.env` desde `.env.example` si no existe
- Construye imagenes y levanta el stack
- Espera healthchecks
- Inicializa la base de Airflow y crea usuario admin
- Crea topics Kafka base
- Ejecuta migraciones SQL en PostgreSQL y TimescaleDB
- Verifica el estado final de servicios

El script es idempotente: puede correrse multiples veces sin efectos destructivos. Esto reduce friccion en entornos nuevos.

### 1.4 Ajustes operativos y fixes tempranos

Durante la estabilizacion se incorporaron mejoras:

- Separacion de bases de datos de Airflow y MLflow para evitar colisiones de metadata.
- Ajuste de bootstrap para evitar deadlocks al iniciar Airflow.
- Restriccion de exposicion de Kafka UI al host.
- Alineacion de `.env.example` con el stack real.

Estos cambios dejaron el entorno listo para iterar sobre bases de datos e ingestion.

## Fase 2: bases de datos

### 2.1 TimescaleDB: series temporales

Se implemento la migracion `database/timescaledb/migrations/001_initial_schema.sql` con los siguientes componentes:

- Tabla `public.transactions` con PK `(transaction_id, timestamp)` para compatibilidad con hypertable.
- Hypertable por columna `timestamp` con chunk diario.
- Indices para consultas por usuario y por tiempo.
- Indice parcial para fraude (optimiza dashboards de fraude).
- Continuous aggregates:
  - `fraud_volume_hourly` (tasa de fraude por hora)
  - `merchant_amount_daily` (monto diario por merchant)
- Politicas:
  - Refresh de cagg cada 5 minutos
  - Compresion despues de 7 dias
  - Retencion y drop despues de 2 anios

Se incluyeron queries de verificacion manual para validar tablas, indices y policies.

### 2.2 PostgreSQL: metadata operacional

Se creo la migracion `database/postgresql/migrations/001_initial_schema.sql` con tablas operativas:

- `model_deployments`: versiones de modelos y metricas.
- `predictions_history`: historico de predicciones y latencia.
- `drift_reports`: reportes de drift por feature.
- `alert_log`: alertas operativas.

Se agregaron constraints y checks para asegurar calidad de datos (rangos de scores, orden temporal, severities permitidas). Tambien se crearon indices para consultas frecuentes por fecha, version y estado.

En `database/postgresql/stored_procedures/001_initial_stored_procedures.sql` se incorporaron funciones:

- `activate_model_version`: activa una version y desactiva el resto.
- `check_fraud_rate`: evalua tasa de fraude y emite alertas.

Finalmente, en `database/postgresql/triggers/001_initial_triggers.sql` se registraron triggers para:

- Alertar sobre tasas de fraude altas en `predictions_history`.

### 2.3 Seeds y soporte de datos

Se agrego un generador de seeds para TimescaleDB en `database/timescaledb/seeds/seed_transactions.py` con guia en `database/timescaledb/seeds/README.md`. El objetivo fue contar con datos sinteticos para pruebas de dashboards y validacion de queries.

## Fase 3: ingesta y streaming con Kafka

### 3.1 Contratos de datos y schemas

Se crearon schemas Avro en `ingestion/schemas` para estandarizar mensajes:

- `transaction_raw.avsc`
- `transaction_features.avsc`
- `transaction_prediction.avsc`
- `fraud_alert.avsc`

Se levanto Schema Registry y se configuro compatibilidad backward. Esto garantiza evolucion controlada de los eventos.

### 3.2 Producer: simulacion realista de transacciones

Se definio el modelo base `Transaction` en `ingestion/models.py` y se implemento un generador de transacciones legitimas en `generator.py`, con:

- Distribuciones log-normales por categoria
- Preferencias por pais y dispositivo
- Sesiones y hashes de IP consistentes
- Sesgos de actividad diaria

Se agrego un generador de fraude (`FraudPatternGenerator`) con cuatro patrones:

- Monto atipico
- Pais inusual
- Rafaga de alta frecuencia
- Merchant desconocido con monto alto

En `ingestion/producer/main.py` se expusieron modos CLI:

- `live`: mezcla de legitimas y fraude a tasa configurable
- `replay`: reproduce CSV historico
- `scenario`: inyecta un patron especifico o mixto

El producer reporta estadisticas y soporta TPS configurable.

### 3.3 Kafka producer con Avro

En `ingestion/producer/kafka_producer.py` se construyo un productor con:

- Serializacion Avro con `fastavro`
- Idempotencia habilitada
- Compresion snappy
- Retries y acks=all
- Logs de entrega y manejo de errores

Esto permite un pipeline robusto desde el inicio.

### 3.4 Consumer base y deserializacion

Se implemento `ingestion/consumer/transaction_consumer.py` para:

- Consumir mensajes Avro de `transactions.raw`
- Convertir a `TransactionRaw`
- Manejar timestamps con timezone
- Retry simple para fallos de deserializacion

El consumer evita autocommit y solo confirma offsets una vez procesada la transaccion.

### 3.5 Feature engineering online

Se construyo el pipeline en `ingestion/consumer/main.py`:

1. Consume transaccion
2. Calcula features de ventana (1h, 24h, 7d)
3. Calcula features historicas por usuario
4. Actualiza estado en memoria
5. Persiste estado en Redis (si disponible)
6. Inserta en TimescaleDB (si disponible)
7. Publica evento enriquecido en `transactions.features`

Los calculos de ventana viven en `window_store.py` y los historicos en `historical_store.py`. Los modelos de features estan en `features.py`.

### 3.6 Redis como feature store

Se agrego `user_store.py` para guardar y rehidratar el estado por usuario:

- Ventanas: `features:window:<user_id>`
- Historico: `features:historical:<user_id>`
- TTL de 7 dias

El consumer hidrata el estado al primer evento de cada usuario, lo que permite continuidad aun tras reinicios.

### 3.7 TimescaleDB writer

En `transaction_store.py` se implemento insercion idempotente en `public.transactions`, con pool de conexiones y manejo de errores. Si Timescale no esta disponible, el consumer sigue procesando el stream sin persistencia, evitando bloqueos.

### 3.8 Publicacion de features

`feature_publisher.py` serializa las features en Avro y publica en `transactions.features`. Se flatean los valores de ventana e historico en un `map<string,double>` para compatibilidad con el schema.

## Fase 4: modelo ML con XGBoost

### 4.1 Exploracion y analisis de datos (EDA)

Se construyeron tres notebooks en `model/notebooks/`:

- `eda_base.ipynb`: distribucion de clases (1-3% de fraude), distribucion de amounts por categoria, analisis de paises y devices.
- `eda_correlations.ipynb`: matriz de correlacion entre features, importancia inicial con modelo simple, identificacion de features candidatas a eliminar.
- `eda_temporal.ipynb`: patrones de fraude por hora del dia, dia de la semana y mes; estacionalidad que puede afectar el modelo.

Los hallazgos quedaron documentados en `docs/eda_findings.md`. El mas relevante para el pipeline offline fue la correlacion perfecta (r = 1.0) entre `tx_velocity_1h` y `tx_count_1h`, que anticipo la decision de eliminar la primera en el paso de seleccion de features.

### 4.2 Feature engineering offline

El objetivo central de esta seccion fue replicar con exactitud el pipeline online (Fase 3) pero en modo batch, sin filtrar data del futuro.

**TransactionFeaturizer** (`offline_features/featurizer.py`) implementa las mismas 18 features del pipeline de streaming usando:

- Binary search y prefix sums para ventanas temporales (1h, 24h, 7d), con complejidad O(n log n) total por usuario.
- Estado incremental O(n) para las features de perfil historico (ratio de monto, paises y merchants nuevos).
- La invariante clave: para la transaccion i solo se usan transacciones con timestamp estrictamente menor, lo que elimina data leakage.

**Encoders** (`offline_features/encoders.py`) implementa `CategoricalEncoderPipeline` con dos estrategias:

- `TargetEncoder` con suavizado (smoothed mean target encoding) para `merchant_category` y `country`. Requiere la variable objetivo en fit para calcular las medias por categoria.
- `OrdinalEncoder` para `device_type`.

Los encoders se persisten como artefactos para garantizar que el serving use exactamente los mismos valores.

**Manejo de desbalance** (`offline_features/imbalance_strategies.py`) evalua dos estrategias: SMOTE (oversampling de la clase fraude) y `scale_pos_weight` de XGBoost. Incluye `ImbalanceReport` con metricas comparativas por estrategia. En la practica se opto por `scale_pos_weight` calculado como `(n_negatives / n_positives)` sobre el set de entrenamiento.

**Seleccion de features** (`offline_features/feature_selection.py`) combina tres metodos:

- Importancia XGBoost basada en gain para descartar features de bajo impacto (umbral: < 1% del gain total).
- Correlacion de Pearson para detectar pares redundantes (umbral: |r| > 0.85). Elimina la feature con menor gain del par.
- Boruta opcional (desactivado por defecto) para confirmacion estadistica.

El resultado quedo fijado en `model/selected_features.py` como `SELECTED_FEATURES`: 16 features finales. `tx_velocity_1h` fue descartada por ser numericamente identica a `tx_count_1h` (r = 1.0). `device_type_encoded` fue descartada por importancia < 1% sobre el dataset de 50 000 filas. Todas las demas features superaron los umbrales de importancia y correlacion.

`TransactionFeaturizer.apply_selection()` permite encadenar la seleccion directamente al featurizer para que `transform()` y `get_feature_names()` respeten el subconjunto elegido.

### 4.3 Entrenamiento del modelo XGBoost

**Script de entrenamiento** (`model/train.py`) orquesta el pipeline completo:

1. Carga transacciones etiquetadas desde TimescaleDB.
2. Aplica `TransactionFeaturizer` (fit en train, transform en val y test).
3. Realiza un split temporal — nunca aleatorio — con 70/15/15 sobre el eje de tiempo para evitar data leakage.
4. Calcula `scale_pos_weight` desde el set de entrenamiento.
5. Entrena XGBoost con parametros base o con los mejores parametros del tuning si se pasa `--tune`.
6. Guarda modelo, encoders y metadata de entrenamiento en `artifacts/model/`.

**MLflow tracking**: cada run registra parametros de XGBoost, metricas del test set (F1, precision, recall, AUC-ROC), el rango temporal del dataset, y los siguientes artefactos: confusion matrix, curva ROC, curva PR, feature importances y analisis de threshold. El modelo se registra en el MLflow Model Registry como `FraudDetectionModel` con stage `Staging`.

Un ajuste operativo necesario fue alinear las versiones de MLflow y XGBoost y exponer el artifact store del servidor para que el registry pudiera leer los artefactos de runs locales de forma confiable.

**Hyperparameter tuning** (`model/tuning.py`) usa Optuna con `TPESampler`. La funcion objetivo maximiza PR-AUC en el set de validacion. El espacio de busqueda incluye `n_estimators`, `max_depth`, `learning_rate`, `min_child_weight`, `subsample`, `colsample_bytree` y `scale_pos_weight`. Cada trial queda loggeado en MLflow. El tuning se activa con la flag `--tune`.

**Evaluacion de negocio** (`model/evaluate.py`, primera iteracion) agrega optimizacion de threshold basada en costo. El threshold por defecto (0.5) es reemplazado por el que minimiza la funcion de costo `FN * 100 + FP * 5`, donde 100 es el costo relativo de un fraude no detectado y 5 el de un falso positivo. El threshold optimo y el costo total por transaccion quedan loggeados como artefactos y metricas en MLflow.

### 4.4 Validacion y promocion de modelo

**Quality gates** (`model/evaluate.py`, segunda iteracion) implementa `run_quality_gates()` con tres umbrales fijos:

- F1-score >= 0.85
- AUC-ROC >= 0.90
- Latencia P99 <= 50ms (medida en batch de 1000 transacciones con 10 repeticiones)

El proceso carga el modelo desde el MLflow Registry por nombre y version, construye el test set temporal desde TimescaleDB (ultimo 20% cronologico), aplica el featurizer y evalua. Los resultados se escriben de vuelta al run de MLflow como metricas de quality gate. Si alguna gate falla, el script termina con exit code 1. Esto lo hace integrable en cualquier pipeline de CI.

**Comparacion challenger vs champion** (`model/evaluate.py`, tercera iteracion) implementa `compare_challenger_vs_champion()`. Carga el modelo en stage `Production` como champion y el challenger por version. Ambos se evaluan sobre el mismo test set temporal. El challenger gana si su F1 supera al champion en mas de 0.02 (2 puntos porcentuales). Si no hay champion en produccion, el challenger gana por defecto. La funcion `--compare` de la CLI ejecuta primero los quality gates y solo procede a la comparacion si todos pasan.

**Promocion a produccion** (`model/promote.py`) implementa `promote_to_production()` con las siguientes garantias:

1. Verifica que el modelo este en stage `Staging` antes de proceder.
2. Obtiene metadata del run de MLflow: metricas (F1, precision, recall, AUC-ROC) y ventana temporal del dataset de entrenamiento.
3. Ejecuta una transaccion PostgreSQL que inserta o reutiliza el registro en `public.model_deployments` por `mlflow_run_id` y llama al stored procedure `activate_model_version` para desactivar versiones anteriores.
4. Solo si la transaccion de base de datos confirma sin errores, transiciona el modelo en MLflow a `Production` con `archive_existing_versions=True`.

Este orden garantiza que la base de datos nunca quede desincronizada del registry: si MLflow falla, la DB no queda con una version activa que no existe; si la DB falla, el modelo no se promueve.

### 4.5 Refactoring y estabilizacion

Durante las pruebas end-to-end de la Fase 4 se realizaron los siguientes ajustes:

- **Consolidacion de ingestion**: `Transaction` se movio a `ingestion/models.py` y utilidades de timezone a `ingestion/utils.py`. `AvroPublisher` se extrajo a `ingestion/kafka_publisher.py`. Los modulos del consumer se renombraron: `windows.py` → `window_store.py`, `historical.py` → `historical_store.py`, `feature_models.py` → `features.py`.
- **Extraccion de modulos del modelo**: funciones de evaluacion a `model/metrics.py`, de visualizacion a `model/plots.py`. `model/features.py` se renombro a `model/selected_features.py`. Se corrigio un bug en `select_features` donde `max_iter` era usado sin estar declarado como parametro.
- **Seed simplificado**: se reemplazo `scripts/seed-timescale.sh` por el comando Docker documentado en `database/timescaledb/seeds/README.md`, corrigiendo ademas los nombres de variables de entorno y agregando `PYTHONPATH=/app`.
- **Dependencias**: se agregaron `joblib`, `psycopg2-binary` y `pydantic-settings` al grupo `model` de `pyproject.toml`.

## Fase 5: Serving e inference

### 5.1 Estructura FastAPI y carga del modelo

Se construyó el módulo `serving/` con FastAPI como framework de inferencia sincrónica. La aplicación se bootstrapea en `serving/app/main.py` con un lifecycle `lifespan` que:

- Carga el modelo desde MLflow Registry vía `ModelLoader` en `serving/app/services/model_loader.py`. Conecta al tracking server, obtiene la última versión en `Production`, descarga artifacts a `/tmp/fraud_model/`, y carga `xgboost_model.joblib` y `categorical_encoder.joblib` con joblib.
- Consulta en PostgreSQL el `deployment_id` activo desde `public.model_deployments`.
- Inicializa un pool asyncpg con tamaño configurable por variable de entorno.
- Inicializa `PredictionCache` contra Redis (degradación graceful si no está disponible).
- Si el modelo no puede cargarse, el servicio arranca en modo **degraded** — el health check lo reporta y `/predict` devuelve 503.

`ModelLoader.prepare_features(raw, window_features)` produce un array numpy de 16 features combinando datos crudos de la transacción (log1p del monto, hora y día de la semana, target encoding de merchant_category y country con fallback a media global) con las 11 features de ventana y perfil histórico recibidas del consumer upstream.

### 5.2 Endpoints de predicción

Se expusieron dos endpoints de inferencia en `serving/app/routes/predict.py`:

- **`POST /predict`**: predicción individual. Recibe el payload de transacción completo incluyendo el mapa `features` con las ventanas computadas upstream. Antes de inferir verifica caché Redis por `transaction_id`. Si hay cache hit, devuelve la respuesta sin ejecutar el modelo. Calcula la latencia desglosada en `feature_ms` (preparación) e `inference_ms` (predict_proba). Persiste asincrónicamente con `BackgroundTasks` vía `PredictionStore`. Publica logs estructurados con alertas para requests lentas (umbral configurable).

- **`POST /predict/batch`**: predicción batch (1 a 500 transacciones). Internamente prepara features individualmente, hace `np.vstack` para inferencia vectorizada con XGBoost, y persiste cada predicción individual con `BackgroundTasks`. Retorna latencia promedio por transacción y latencia total del batch.

Los schemas Pydantic están en `serving/app/schemas/prediction.py`: `TransactionRequest`, `PredictionResponse`, `BatchPredictionRequest`, `BatchPredictionResponse`. Todos con validación (amount > 0, batch size entre 1 y 500).

### 5.3 Persistencia y caché

**PredictionStore** (`serving/app/services/prediction_store.py`): persiste cada predicción en `public.predictions_history` usando `asyncpg` con un pool de conexiones. Opera en modo fire-and-forget mediante `BackgroundTasks` de FastAPI para no bloquear la respuesta HTTP.

**PredictionCache** (`serving/app/services/cache.py`): caché en Redis con socket_timeout de 100ms y TTL de 60s por clave (`prediction:<transaction_id>`). Si Redis no está disponible al iniciar, se degrada gracefulmente y desactiva la caché sin fallar. Los errores de Redis en runtime se loggean como warning.

**Middleware de timing** (`serving/app/main.py`): agrega el header `X-Process-Time-Ms` a cada respuesta con la latencia total del request.

### 5.4 Inference consumer: del stream a la API

Se implementó un nuevo consumer en `ingestion/inference_consumer/` que cierra el loop online:

**FeatureConsumer** (`consumer.py`): consume mensajes Avro del topic `transactions.features`. Deserializa con fastavro, maneja errores con retry queue (un reintento por mensaje), y commit manual solo tras inferencia exitosa.

**InferenceApiClient** (`api_client.py`): cliente HTTP sincrónico con httpx que llama a `POST /predict` de FastAPI. Timeout de 2s. Convierte UTC el timestamp antes de enviar.

**CircuitBreaker** (`circuit_breaker.py`): breaker en memoria con tres estados (CLOSED → OPEN → HALF_OPEN). Abre tras 5 fallos consecutivos y reintenta tras 30s de cooldown. Cuando está OPEN, salta inferencia sin llamar a la API.

**PredictionPublisher** (`prediction_publisher.py`): serializa en Avro el resultado de la predicción y lo publica en `transactions.predictions` usando `AvroPublisher`.

**AlertPublisher** (`alert_publisher.py`): cuando el label de predicción es positivo, clasifica la severidad (`CRITICAL` si score >= 0.90, `HIGH` si >= 0.75, `WARNING` en otro caso) y publica una alerta en `fraud.alerts`.

**Orquestación** (`main.py`): loop principal que consume, infiere, publica predicción, publica alerta si corresponde, y commitea el offset. Todo con señales SIGINT/SIGTERM para shutdown graceful. Soporta rate limiting configurable entre requests.

### 5.5 Endpoints de health y modelo

En `serving/app/routes/health.py`:

- **`GET /health`**: retorna siempre 200 con `{"status": "ok" | "degraded", "model_loaded": true | false}`. El estado `degraded` indica que FastAPI corre sin modelo cargado — el servicio igual responde en /health pero /predict devuelve 503.
- **`GET /model/info`**: retorna metadata del modelo activo (nombre, versión, stage, deployment_id, fraud_score_threshold). Devuelve 503 si el modelo no está cargado.

### 5.6 Ajustes operativos y Docker

Se realizaron varios fixes para estabilizar el serving:

- El healthcheck del contenedor Docker se simplificó a TCP connect en puerto 8000 para evitar dependencia del modelo.
- El directorio `/tmp` se crea explícitamente antes de arrancar para que el proceso no-root pueda escribir artifacts.
- `config.py` se agregó al COPY del Dockerfile de FastAPI para resolver imports.
- El directorio `offline_features/` se incluyó en la imagen de serving para que los encoders puedan aplicarse correctamente.
- Se parametrizaron `min_size` y `max_size` del pool asyncpg, y `workers` de uvicorn, vía variables de entorno.

### 5.7 Dependencias y documentación

Se regeneró `uv.lock` con las nuevas dependencias: `asyncpg` y el grupo `inference` para el consumer. Se actualizó `docs/glossary.md` con los términos de serving e inference. Se documentó la arquitectura completa en `serving/README.md` con diagrama de flujo, ejemplos de request/response y descripción de cada servicio interno.

## Fase 6: MLOps pipeline — reentrenamiento, drift y calidad de datos

### 6.1 Inicialización de MLflow

Se creó `mlops/mlflow/init_mlflow.py` como script idempotente de bootstrap del servidor MLflow. Crea el experimento `fraud-detection-v1` si no existe, le asigna tags de proyecto (algoritmo, tarea, versión de datos) y configura los metadatos del modelo registrado `FraudDetectionModel` en el Registry (descripción del ciclo de vida, tags de serving y selección de features). El script reintenta la conexión hasta cinco veces antes de fallar, lo que lo hace seguro para ejecutarse desde el setup general incluso cuando MLflow demora en arrancar.

Este script se integró en `scripts/setup.sh` como paso explícito tras el healthcheck de MLflow, garantizando que el experimento y el registro existan antes de que cualquier run de entrenamiento intente loggearse.

### 6.2 DAGs de Airflow

Se construyeron cuatro DAGs en `mlops/airflow/dags/` que cubren el ciclo de vida completo del modelo en producción.

**`retrain_fraud_model`** corre diariamente a las 02:00. Verifica que haya al menos 1000 transacciones etiquetadas en TimescaleDB antes de proceder. Si el dato es suficiente, invoca `model/train.py` como subproceso capturando su salida para extraer el `run_id` de MLflow. Al terminar el entrenamiento, localiza la última versión en `Staging` y dispara el DAG `validate_and_promote_model` pasándole `model_name`, `model_version` y `run_id` vía `dag_run.conf`. Si los datos no alcanzan el mínimo, el DAG hace `AirflowSkipException` para no generar runs fallidos innecesarios.

**`validate_and_promote_model`** es event-driven (schedule `None`): solo corre cuando lo dispara `retrain_fraud_model` o una intervención manual. Recibe los parámetros del modelo vía `dag_run.conf` y ejecuta en secuencia:

1. `run_quality_gates_task`: invoca `model.evaluate.run_quality_gates()` — F1 ≥ 0.85, AUC-ROC ≥ 0.90, latencia P99 ≤ 50 ms.
2. `compare_with_champion`: si los gates pasan, invoca `model.evaluate.compare_challenger_vs_champion()`. El challenger debe superar al champion por más de 0.02 de F1. Si no hay champion en `Production`, el challenger gana por defecto.
3. `promote_to_production_task`: llama a `model.promote.promote_to_production()`, que transacciona la BD y MLflow atomicamente.
4. `archive_rejected_version`: con `TriggerRule.ONE_FAILED` — si alguno de los pasos anteriores falla o hace skip, archiva la versión en MLflow para evitar acumulación de versiones huérfanas en `Staging`.

**`data_quality_check`** corre cada hora. Ejecuta tres checks en paralelo:

- `check_transaction_volume`: cuenta transacciones en la última hora en TimescaleDB. Si están por debajo del umbral configurable `MIN_TRANSACTIONS_PER_HOUR` (default: 10), inserta una alerta `LOW_TRANSACTION_VOLUME` en `alert_log`.
- `check_prediction_rate`: análogo para `predictions_history` en PostgreSQL. Umbral: `MIN_PREDICTIONS_PER_HOUR` (default: 5).
- `check_amount_distribution`: evalúa distribución de montos en las últimas 24 horas. Alerta si el promedio cae por debajo de 1.0 o si alguna transacción supera 100 000.

**`drift_detection_report`** corre cada seis horas. Es el DAG más complejo y coordina varios tipos de análisis en paralelo:

1. Obtiene el deployment activo desde `model_deployments`.
2. Descarga el dataset de referencia (`reference_dataset.parquet`) del artefacto MLflow del run de entrenamiento (`mlops/evidently/reference.py`).
3. Extrae datos de producción de las últimas 24 horas desde TimescaleDB con `TimescaleExtractOperator`.
4. Featuriza ambos datasets con `TransactionFeaturizer` usando los encoders del run activo.
5. Ejecuta `EvidentlyReportOperator` (data drift) y `run_model_drift_task` (model drift) en paralelo.
6. Persiste el reporte en `drift_reports` y evalúa las acciones correctivas.
7. Exporta el reporte HTML de Evidently como artefacto al run MLflow activo.

### 6.3 Operadores custom de Airflow

Se implementaron tres operadores reutilizables en `mlops/airflow/plugins/fraud_operators.py`:

**`TimescaleExtractOperator`**: consulta TimescaleDB (o PostgreSQL) con SQL arbitrario y serializa el resultado a un archivo Parquet en una ruta configurable. Devuelve la ruta como XCom para encadenar con tareas downstream. Acepta `conn_settings_fn` para apuntar a distintas bases.

**`MLflowRegisterModelOperator`**: lee el `model_version` desde XCom de un task previo y transiciona el modelo a un stage en el Registry. Soporta `archive_existing_versions` para limpiar automáticamente versiones anteriores en el mismo stage.

**`EvidentlyReportOperator`**: lee los paths de Parquet desde XCom de los tasks de featurización y ejecuta `run_data_drift_report()`. Devuelve un dict con `drift_score`, `dataset_drift` y `feature_drifts` por columna, que queda disponible como XCom para tareas de persistencia y alertas.

### 6.4 Detección de drift con Evidently AI

Se construyó el módulo `mlops/evidently/` con responsabilidades separadas por archivo.

**`data_drift.py`** encapsula la ejecución del `DataDriftPreset` de Evidently. La función `_build_and_run_report()` es privada y corre el reporte completo; las funciones públicas `run_data_drift_report()` y `run_data_drift_report_with_html()` exponen dos modos: solo métricas (para la mayoría de los casos) y métricas más HTML (para los reportes periódicos). El resultado se estructura en `DataDriftResult` con `drift_share`, `dataset_drift` y un dict de `FeatureDriftResult` por feature.

**`model_drift.py`** mide degradación de performance del modelo en producción. `fetch_labeled_predictions()` extrae de PostgreSQL las predicciones que tienen `actual_label` marcado (transacciones con feedback real) para el deployment activo en los últimos 7 días. `run_model_drift_report()` calcula F1, precisión y recall actuales con scikit-learn y los compara contra las métricas de referencia almacenadas en `model_deployments`. Si el F1 cae más de 0.05 puntos respecto al baseline, `drift_detected` es `True`. Requiere al menos 50 predicciones etiquetadas para ser concluyente; si no hay suficiente dato, retorna `has_sufficient_data=False` sin fallar.

**`thresholds.py`** centraliza los umbrales de alerta y la lógica de severidad. Define tres umbrales configurables por variable de entorno: `DRIFT_THRESHOLD_CRITICAL` (0.20), `DRIFT_THRESHOLD_GLOBAL` (0.30) y `MODEL_F1_DEGRADATION_THRESHOLD` (0.05). El set `CRITICAL_FEATURES` identifica las cinco features con mayor impacto operativo. `evaluate_drift_action()` combina los resultados de data drift y model drift en una `DriftAction` con severidad (`INFO`, `WARNING`, `HIGH`, `CRITICAL`) y decide si disparar reentrenamiento. Solo los niveles `HIGH` y `CRITICAL` activan el DAG de reentrenamiento vía la REST API de Airflow.

**`drift_store.py`** implementa `DriftReportStore` con dos métodos: `save()` persiste el reporte completo en `public.drift_reports` serializando los feature drifts como JSONB, y `save_alert()` inserta en `public.alert_log`. Se diseñó como clase separada para que el DAG pueda intercambiarla por un stub en tests sin modificar la lógica del pipeline.

**`reference.py`** carga el dataset de referencia desde MLflow: descarga `reference_dataset.parquet` del artefacto del run de entrenamiento y lo retorna como DataFrame. Esto garantiza que el drift siempre se calcule contra exactamente los datos con los que se entrenó la versión activa, no contra un archivo estático que podría desincronizarse.

**`html_exporter.py`** sube el HTML generado por Evidently al artifact store de MLflow usando `MlflowClient.log_artifact()`. Si la subida falla (por ejemplo, si el artifact store es de solo lectura), loggea un warning y retorna `None` sin interrumpir el DAG, ya que el reporte en PostgreSQL ya fue persistido.

### 6.5 Ciclo de vida completo del modelo

La combinación de los cuatro DAGs crea un ciclo de vida autónomo:

```
[TimescaleDB] ──► retrain_fraud_model (diario 02:00)
                       │
                       ▼
              validate_and_promote_model (event-driven)
                       │
              ┌────────┴────────┐
              │ quality gates   │ champion comparison
              │ pasan           │ challenger gana
              └────────┬────────┘
                       ▼
              promote_to_production → [PostgreSQL] + [MLflow Registry]
                       │
              ┌────────┴────────────────────────────────────┐
              │                                             │
    drift_detection_report (c/6h)          data_quality_check (c/1h)
              │                                             │
    ┌─────────┴─────────┐                              [alert_log]
    │ data drift        │ model drift
    │ (Evidently)       │ (sklearn vs. baseline)
    └─────────┬─────────┘
              ▼
    DriftAction: HIGH/CRITICAL
              │
              ▼
    trigger_retrain_dag → retrain_fraud_model
```

Esta arquitectura garantiza que ningún paso del ciclo depende de intervención manual: el reentrenamiento dispara la validación, la validación promueve o archiva, el drift detectado vuelve a disparar el reentrenamiento.

## Fase 7: monitoreo y observabilidad

### 7.1 Instrumentación de métricas con Prometheus

El objetivo de esta fase fue dar visibilidad operativa al stack en tiempo real, complementando los reportes de drift (que trabajan sobre datos históricos) con métricas de infraestructura y comportamiento HTTP.

**Datasource Prometheus en Grafana** (`monitoring/grafana/provisioning/datasources.yaml`): se agregó un tercer datasource al archivo de provisioning con uid fijo `prometheus`, apuntando a `http://prometheus:9090`. Se eligió `httpMethod: POST` porque Grafana 10.4+ lo recomienda para queries largas. El uid fijo permite que los dashboards JSON lo referencien sin depender de IDs autogenerados por la UI.

**Instrumentación de FastAPI** (`serving/app/main.py`): se integró `prometheus-fastapi-instrumentator` con una sola línea al nivel del módulo: `Instrumentator().instrument(app).expose(app)`. Esto expone automáticamente el endpoint `GET /metrics` con métricas estándar: `http_requests_total` (contador por método, handler y status code), `http_request_duration_seconds` (histograma de latencia con buckets) y `up` (liveness del target). La inicialización va fuera del lifespan porque debe ejecutarse en tiempo de importación del módulo, no en el arranque del servidor.

La dependencia se declaró en `pyproject.toml` como `prometheus-fastapi-instrumentator>=0.9.0` (sin upper bound), lo que resolvió un problema de compatibilidad: el constraint original `<1.0.0` no tenía wheels disponibles para Python 3.14, que forma parte del rango `requires-python = ">=3.11"` del proyecto. El lockfile se regeneró con `uv lock` y resolvió en v7.1.0.

**Servicio Prometheus** (`docker-compose.yml`): se agregó `prom/prometheus:v2.51.0` con un volumen bind-mount de solo lectura para la configuración y un volumen named `prometheus-data` para la retención de series temporales. Los flags de arranque configuran retención a 15 días y habilitan el lifecycle endpoint (permite recargar configuración sin reiniciar). El servicio Grafana incorporó `prometheus: condition: service_started` en su `depends_on`.

**Configuración de scrape** (`monitoring/prometheus/prometheus.yml`): un único job `fastapi` que scrapea `fastapi:8000/metrics` cada 10 segundos. El scrape global se fijó en 15 segundos. No se incluyeron exporters adicionales (kafka-exporter, node-exporter) porque habrían requerido servicios extra no contemplados en el stack; esta limitación queda documentada en el Panel 10 del dashboard de system health.

### 7.2 Dashboards de Grafana

Se implementaron cuatro dashboards provisionados automáticamente mediante archivos JSON en `monitoring/grafana/dashboards/`. El Dockerfile de Grafana los copia a `/etc/grafana/dashboards/` (fuera del volumen `grafana-data`) y el archivo de provisioning apunta a esa ruta. Esta decisión evita que el volumen named, que persiste entre reinicios y puede contener datos de versiones anteriores, tape los archivos de la imagen con versiones desactualizadas. El provisioning de dashboards apunta en `monitoring/grafana/provisioning/dashboards.yaml`.

Todos los dashboards usan `schemaVersion: 39` y quedan bajo la carpeta **Fraud Detection** de Grafana.

**`fraud_alerts.json` — Alertas en vivo** (uid: `fraud-alerts`, refresh 30s, ventana 1h): seis paneles que combinan el datasource `timescaledb` (gauge de tasa de fraude actual desde `fraud_volume_hourly`, time series de volumen por minuto, tabla de top 10 merchants, dos stat de conteos horarios) y el datasource `postgresql` (tabla de alertas recientes desde `alert_log`). La tabla de alertas aplica `color-background` sobre el campo `Severidad` con value mappings por texto para colorear CRITICAL/HIGH/WARNING/INFO sin depender de valores numéricos.

**`model_performance.json` — Model Performance** (uid: `model-performance`, refresh 5m, ventana 30d): siete paneles exclusivamente sobre el datasource `postgresql`. Los cuatro stat de métricas del modelo activo (F1, AUC-ROC, precisión, recall) usan `percentunit` y thresholds alineados con los quality gates del sistema (F1 ≥ 0.85, AUC-ROC ≥ 0.90). El time series de latencia usa `to_timestamp(floor(extract(epoch FROM col) / 300) * 300)` para bucketing de 5 minutos compatible con PostgreSQL estándar, sin `time_bucket`. La tabla champion vs. challenger colorea las columnas F1 y AUC-ROC con los mismos umbrales para facilitar la comparación visual. El histograma de distribución de scores delega el bucketing al motor de Grafana, pasando los valores crudos con `prediction_score` como serie.

**`drift_monitor.json` — Data Drift Monitor** (uid: `drift-monitor`, refresh 10m, ventana 7d): siete paneles sobre `postgresql`. El panel de estado del modelo (semáforo) usa `value mappings` por texto (OK → verde, WARNING → amarillo, DRIFT → rojo) en lugar de thresholds numéricos, porque la columna `Estado` es una cadena generada por un `CASE WHEN` en la query. Las líneas de referencia del time series de evolución del drift score (umbral crítico 0.20 y global 0.30) se implementan con `thresholdsStyle: line` en el field config, que dibuja líneas horizontales automáticas sobre la serie. El bar chart de drift por feature expande la columna JSONB `feature_drifts` de `drift_reports` con `jsonb_each()` directamente en la query SQL, devolviendo una fila por feature del último reporte.

**`system_health.json` — System Health** (uid: `system-health`, refresh 30s, ventana 1h): diez paneles que cruzan los tres datasources. Los seis primeros usan PromQL sobre `prometheus`: stat de up/down con value mappings 0→DOWN/1→UP, request rate con `round(sum(rate(...)))`, error rate de 5xx como ratio, gauge de P99 con thresholds en 50ms y 200ms, y dos time series de rate por endpoint y latencia por percentil. Los paneles 7 y 8 son stat de conteos sobre `postgresql` y `timescaledb` respectivamente. El panel 9 usa el datasource especial `-- Mixed --` (uid `-- Mixed --`) para combinar en un único time series las transacciones (`time_bucket` en TimescaleDB) y las predicciones (`to_timestamp(floor(...))` en PostgreSQL), con cada query especificando su datasource propio en el campo `datasource` del target. El panel 10 es un panel `text` con Markdown que documenta la ausencia de Kafka consumer lag, indicando qué servicio adicional sería necesario.

### 7.3 Alertas provisionadas con Unified Alerting

Se creó `monitoring/grafana/provisioning/alerting/alerts.yaml` con cuatro reglas de alerta bajo el grupo `fraud-detection-alerts`, carpeta `Fraud Detection`, evaluadas cada minuto.

Cada regla sigue el pipeline estándar de Grafana Unified Alerting con tres refIds: `A` (query al datasource), `B` (expresión `reduce` con función `last` sobre A) y `C` (expresión `threshold` sobre B que define la condición de disparo). Este diseño desacopla la obtención del dato de la lógica de comparación, lo que permite cambiar umbrales sin modificar queries SQL o PromQL.

Las cuatro reglas:

| Regla | Datasource | Condición | `for` | Severidad |
|---|---|---|---|---|
| High Fraud Rate | `timescaledb` | `fraud_rate * 100 > 5` | 10m | critical |
| FastAPI P99 Latency High | `prometheus` | `histogram_quantile P99 > 200ms` | 5m | warning |
| Data Drift Detected | `postgresql` | `drift_score > 0.30` | 0s | critical |
| Fraud Spike — 5 min | `timescaledb` | `COUNT(is_fraud) > 5` en 5 min | 0s | high |

Un bug encontrado durante la integración fue que Grafana 10.4 rechaza reglas sin `relativeTimeRange` explícito en cada entrada del array `data`, fallando al arrancar con `invalid relative time range: {From:0s To:0s}`. Se corrigió agregando `from`/`to` en segundos a cada refId: 3600 para queries SQL de ventana horaria, 600 para queries Prometheus de ventana de 10 minutos, y `{from: 0, to: 0}` para los refIds de expresión (que no tienen rango propio).

El Dockerfile de Grafana copia el directorio de alerting al provisioning con `COPY monitoring/grafana/provisioning/alerting/ /etc/grafana/provisioning/alerting/`, y el servicio tiene `GF_UNIFIED_ALERTING_ENABLED=true` y `GF_ALERTING_ENABLED=false` para desactivar el sistema de alertas legado.

### 7.4 Fixes de integración y lecciones

Durante la ejecución de `scripts/setup.sh` se detectaron tres problemas que derivaron en correcciones permanentes:

**Lockfile desactualizado**: al agregar `prometheus-fastapi-instrumentator` a `pyproject.toml` sin correr `uv lock` después, el Dockerfile de FastAPI usa `uv sync --frozen` que lee el lockfile estrictamente. El paquete no estaba en el lockfile, no se instaló en la imagen, y el contenedor fallaba al arrancar con `ModuleNotFoundError`. La corrección exige el paso `uv lock` como parte del workflow de agregar dependencias.

**Volumen Docker tapa el contenido de la imagen**: los dashboards se copiaban a `/var/lib/grafana/dashboards/` en el Dockerfile, pero el servicio monta `grafana-data:/var/lib/grafana`. Si el volumen ya existe (de ejecuciones previas), su contenido prevalece sobre el de la imagen y los archivos nuevos nunca se ven. La solución es copiar los archivos provisionados a rutas que no están cubiertas por ningún volumen. En este caso, `/etc/grafana/dashboards/` es la ruta correcta para dashboards estáticos provisionados.

**healthcheck de Prometheus en setup.sh**: se agregó la función `check_prometheus` (que llama a `/-/healthy`) y los dos `wait_for_service "prometheus"` correspondientes (etapas 3 y 5), junto con la URL `http://localhost:9090` en el resumen final.
