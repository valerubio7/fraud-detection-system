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

### 2.2 PostgreSQL: metadata y auditoria

Se creo la migracion `database/postgresql/migrations/001_initial_schema.sql` con tablas operativas:

- `model_deployments`: versiones de modelos y metricas.
- `predictions_history`: historico de predicciones y latencia.
- `drift_reports`: reportes de drift por feature.
- `alert_log`: alertas operativas.
- `audit_log`: auditoria de cambios.

Se agregaron constraints y checks para asegurar calidad de datos (rangos de scores, orden temporal, severities permitidas). Tambien se crearon indices para consultas frecuentes por fecha, version y estado.

En `database/postgresql/stored_procedures/001_initial_stored_procedures.sql` se incorporaron funciones:

- `activate_model_version`: activa una version y desactiva el resto, con registro en audit_log.
- `calculate_model_metrics`: calcula precision/recall/f1 desde `predictions_history`.
- `check_fraud_rate`: evalua tasa de fraude y emite alertas.
- `audit_trigger`: genera eventos de auditoria para INSERT/UPDATE/DELETE.

Finalmente, en `database/postgresql/triggers/001_initial_triggers.sql` se registraron triggers para:

- Alertar sobre tasas de fraude altas en `predictions_history`.
- Auditar cambios en `model_deployments` y `predictions_history`.

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

Se definio el modelo base `Transaction` en `ingestion/producer/models.py` y se implemento un generador de transacciones legitimas en `generator.py`, con:

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

Se implemento `ingestion/consumer/kafka_consumer.py` para:

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

Los calculos de ventana viven en `windows.py` y los historicos en `historical.py`. Los modelos de features estan en `feature_models.py`.

### 3.6 Redis como feature store

Se agrego `redis_store.py` para guardar y rehidratar el estado por usuario:

- Ventanas: `features:window:<user_id>`
- Historico: `features:historical:<user_id>`
- TTL de 7 dias

El consumer hidrata el estado al primer evento de cada usuario, lo que permite continuidad aun tras reinicios.

### 3.7 TimescaleDB writer

En `timescale_writer.py` se implemento insercion idempotente en `public.transactions`, con pool de conexiones y manejo de errores. Si Timescale no esta disponible, el consumer sigue procesando el stream sin persistencia, evitando bloqueos.

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

**TransactionFeaturizer** (`feature_engineering/offline/featurizer.py`) implementa las mismas 18 features del pipeline de streaming usando:

- Binary search y prefix sums para ventanas temporales (1h, 24h, 7d), con complejidad O(n log n) total por usuario.
- Estado incremental O(n) para las features de perfil historico (ratio de monto, paises y merchants nuevos).
- La invariante clave: para la transaccion i solo se usan transacciones con timestamp estrictamente menor, lo que elimina data leakage.

**Encoders** (`feature_engineering/offline/encoders.py`) implementa `CategoricalEncoderPipeline` con dos estrategias:

- `TargetEncoder` con suavizado (smoothed mean target encoding) para `merchant_category` y `country`. Requiere la variable objetivo en fit para calcular las medias por categoria.
- `OrdinalEncoder` para `device_type`.

Los encoders se persisten como artefactos para garantizar que el serving use exactamente los mismos valores.

**Manejo de desbalance** (`feature_engineering/offline/imbalance.py`) evalua dos estrategias: SMOTE (oversampling de la clase fraude) y `scale_pos_weight` de XGBoost. Incluye `ImbalanceReport` con metricas comparativas por estrategia. En la practica se opto por `scale_pos_weight` calculado como `(n_negatives / n_positives)` sobre el set de entrenamiento.

**Seleccion de features** (`feature_engineering/offline/selection.py`) combina tres metodos:

- Importancia XGBoost basada en gain para descartar features de bajo impacto (umbral: < 1% del gain total).
- Correlacion de Pearson para detectar pares redundantes (umbral: |r| > 0.85). Elimina la feature con menor gain del par.
- Boruta opcional (desactivado por defecto) para confirmacion estadistica.

El resultado quedo fijado en `model/features.py` como `SELECTED_FEATURES`: 17 features finales. `tx_velocity_1h` fue descartada por ser numericamente identica a `tx_count_1h` (r = 1.0). Todas las demas features superaron los umbrales de importancia y correlacion sobre el dataset seed.

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
