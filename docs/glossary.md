# Glosario de términos técnicos

Referencia rápida para el equipo. Los términos están agrupados por área temática.

---

## Infraestructura

**Docker**
Herramienta que empaqueta una aplicación con todo lo que necesita para correr (código, dependencias, configuración) en un contenedor aislado. Garantiza que el sistema se comporte igual en cualquier máquina.

**Docker Compose**
Extensión de Docker que permite levantar varios contenedores a la vez con un solo comando (`docker compose up`). En este proyecto levanta todos los servicios: Kafka, bases de datos, MLflow, Airflow, Grafana, etc.

**uv**
Gestor de dependencias Python moderno. Reemplaza a `pip` y `virtualenv`. Se usa para instalar los paquetes del proyecto y ejecutar scripts con `uv run`.

**Pydantic**
Librería Python para validar datos. En este proyecto se usa para leer y validar las variables de entorno (contraseñas, hosts, puertos) al iniciar cada servicio.

**FastAPI**
Framework Python para construir APIs REST de alto rendimiento. En este proyecto expone el endpoint de scoring: recibe una transacción y devuelve el score de fraude del modelo. Accesible en `http://localhost:8000`.

**Grafana**
Plataforma de visualización y monitoreo. Muestra dashboards con métricas operativas del sistema: volumen de transacciones, tasa de fraude, latencia del modelo, etc. Accesible en `http://localhost:3000`.

**Kafka UI**
Interfaz web para inspeccionar Kafka: ver topics, mensajes, consumer groups y offsets. Útil para debuggear el pipeline de streaming. Accesible en `http://localhost:8080`.

**Healthcheck**
Verificación automática de que un servicio está funcionando correctamente. Docker reinicia el contenedor si el healthcheck falla. Cada servicio del stack tiene uno configurado.

**Idempotente**
Operación que puede ejecutarse múltiples veces sin efectos distintos al primer intento. En este proyecto aplica al script de setup (puede correrse sin romper nada) y a las inserciones en TimescaleDB (no genera duplicados).

---

## Streaming y mensajería

**Kafka**
Plataforma de mensajería distribuida. Actúa como un bus central: los productores publican eventos y los consumidores los leen de forma independiente. En este proyecto transporta transacciones en tiempo real.

**Topic**
Canal de mensajes dentro de Kafka. Cada tipo de evento tiene su propio topic. Los topics del proyecto son: `transactions.raw`, `transactions.features`, `transactions.predictions`, `transactions.fraud.alerts`.

**Producer**
Servicio que genera y publica mensajes a Kafka. En este proyecto simula transacciones bancarias (legítimas y fraudulentas).

**Consumer**
Servicio que lee mensajes de Kafka y los procesa. En este proyecto hay dos consumers: el de features (calcula features y las guarda en base de datos) y el de inferencia (llama a la API de scoring y publica predicciones).

**Inference Consumer**
Consumer especializado que lee mensajes del topic `transactions.features`, llama al endpoint `POST /predict` de la FastAPI y publica el resultado en `transactions.predictions`. Si la predicción indica fraude, también publica en `transactions.fraud.alerts`.

**Avro**
Formato de serialización de datos (similar a JSON pero más compacto y con schema). Todos los mensajes de Kafka en este proyecto usan Avro.

**Schema Registry**
Servicio que almacena y valida los schemas Avro. Garantiza que producer y consumer siempre hablen el mismo "idioma" al intercambiar mensajes.

---

## Bases de datos

**TimescaleDB**
Base de datos especializada en series temporales, construida sobre PostgreSQL. Se usa para almacenar el historial de transacciones con alta eficiencia para consultas por rango de fechas.

**Hypertable**
Tabla especial de TimescaleDB que se particiona automáticamente por tiempo (en este proyecto, por día). Permite consultas muy rápidas sobre rangos temporales grandes.

**Continuous Aggregate**
Vista pre-calculada en TimescaleDB que se actualiza automáticamente. Evita recalcular agregaciones costosas en cada consulta. En este proyecto hay agregados de tasa de fraude por hora y monto por merchant por día.

**PostgreSQL**
Base de datos relacional clásica. En este proyecto almacena metadata operativa: versiones de modelos, historial de predicciones, reportes de drift y logs de auditoría.

**Redis**
Base de datos en memoria (muy rápida). Se usa como feature store temporal: guarda el estado de cada usuario (historial de transacciones) para que el consumer pueda calcular features sin consultar la base de datos principal en cada evento.

**TTL (Time To Live)**
Tiempo de vida de un dato en Redis. Pasado ese tiempo, Redis lo elimina automáticamente. En este proyecto el estado de cada usuario tiene un TTL de 7 días: si un usuario no hace transacciones por más de una semana, su estado se descarta.

**Stored Procedure**
Función almacenada directamente en la base de datos PostgreSQL. En este proyecto `activate_model_version` desactiva versiones anteriores del modelo y activa la nueva dentro de una sola transacción atómica.

**Trigger**
Función que PostgreSQL ejecuta automáticamente cuando ocurre un evento (INSERT, UPDATE, DELETE). En este proyecto dispara alertas cuando la tasa de fraude supera cierto umbral y registra cambios en el log de auditoría.

**Drift**
Degradación del modelo a lo largo del tiempo porque los datos del mundo real cambian. Por ejemplo, si aparece un nuevo patrón de fraude que el modelo nunca vio en entrenamiento, su performance cae. La base de datos tiene una tabla `drift_reports` que almacena cada reporte generado por el pipeline de Evidently.

---

## Machine Learning

**Feature**
Variable de entrada que usa el modelo para tomar decisiones. Ejemplos: monto de la transacción, cantidad de transacciones en la última hora, si el país es nuevo para el usuario.

**Ventana temporal (time window)**
Rango de tiempo hacia atrás desde el momento de una transacción. En este proyecto se calculan features para ventanas de 1 hora, 24 horas y 7 días: por ejemplo, cuántas transacciones hizo el usuario en la última hora.

**Feature Engineering**
Proceso de transformar datos crudos en features útiles para el modelo. En este proyecto incluye cálculos de ventanas temporales, ratios históricos y codificación de categorías.

**Feature Store**
Almacén centralizado de features. Redis actúa como feature store online (tiempo real) y TimescaleDB como feature store offline (entrenamiento).

**XGBoost**
Algoritmo de machine learning basado en árboles de decisión, muy eficiente para datos tabulares. Es el algoritmo que usa este proyecto para detectar fraude.

**Entrenamiento (training)**
Proceso en el que el modelo aprende patrones a partir de datos históricos etiquetados (transacciones con `is_fraud = true/false`).

**Split temporal**
División del dataset en train/validación/test respetando el orden cronológico. Es crítico en fraude para evitar que el modelo "vea el futuro" durante el entrenamiento.

**Data leakage**
Error en el que el modelo accede a información que no estaría disponible en producción (por ejemplo, usar datos futuros para predecir el pasado). Puede inflar artificialmente las métricas.

**Threshold (umbral de clasificación)**
Valor entre 0 y 1. Si el score del modelo supera el threshold, la transacción se marca como fraude. Un threshold bajo detecta más fraude pero genera más falsos positivos.

**Scale_pos_weight**
Parámetro de XGBoost para manejar datasets desbalanceados. Como el fraude es ~2% del total, este valor le da más peso a los casos de fraude durante el entrenamiento.

**SMOTE**
Técnica alternativa para manejar desbalance: genera ejemplos sintéticos de la clase minoritaria (fraude). En este proyecto se evaluó pero se optó por `scale_pos_weight`.

**Target Encoding**
Técnica para convertir variables categóricas (como el país o la categoría del merchant) en números, usando la tasa promedio de fraude de cada categoría.

**Boruta**
Algoritmo de selección de features que determina estadísticamente cuáles son relevantes. Está disponible en el proyecto pero desactivado por defecto por su costo computacional.

**Optuna**
Framework para optimizar hiperparámetros automáticamente. Prueba distintas combinaciones de parámetros de XGBoost y elige la que da mejores resultados. Se activa con `--tune`.

---

## Serving del modelo

**Model Serving**
Proceso de exponer un modelo entrenado como un servicio accesible por otros sistemas. En este proyecto la FastAPI recibe una transacción, ejecuta la inferencia y devuelve el score de fraude en tiempo real.

**Lifespan (FastAPI)**
Mecanismo de FastAPI para ejecutar lógica de startup y shutdown del servidor. En este proyecto se usa para cargar el modelo desde MLflow, crear el pool de conexiones a PostgreSQL e inicializar el cliente de Redis antes de aceptar requests.

**Hot path**
La ruta de código que se ejecuta en cada request de inferencia. Es crítica porque su latencia determina el tiempo de respuesta de la API. En este proyecto incluye: lectura de caché Redis → cálculo de features → inferencia XGBoost → respuesta al cliente.

**BackgroundTasks (FastAPI)**
Mecanismo de FastAPI para ejecutar trabajo después de enviar la respuesta al cliente. En este proyecto se usa para escribir cada predicción en PostgreSQL sin que esa escritura afecte la latencia del endpoint.

**Connection pool (asyncpg)**
Conjunto de conexiones a PostgreSQL mantenidas abiertas y reutilizables entre requests. Evita el overhead de abrir y cerrar una conexión TCP en cada escritura. En este proyecto el pool tiene entre 2 y 10 conexiones, configurables vía variables de entorno.

**asyncpg**
Driver Python asíncrono para PostgreSQL. Mucho más eficiente que psycopg2 en aplicaciones async porque no bloquea el event loop mientras espera respuesta de la base de datos.

**Cache de predicciones**
Respuesta almacenada en Redis indexada por `transaction_id`. Si el mismo `transaction_id` llega dos veces al endpoint `/predict`, la segunda llamada devuelve la respuesta cacheada sin ejecutar el modelo. TTL de 60 segundos.

**Circuit Breaker**
Patrón de resiliencia que protege un servicio de llamadas repetidas a un dependiente caído. Tiene tres estados: CLOSED (operación normal), OPEN (dependiente no disponible, llamadas bloqueadas) y HALF_OPEN (prueba si el dependiente se recuperó). En este proyecto protege al inference consumer ante saturación o caída de la FastAPI.

**Backpressure**
Mecanismo para que un consumidor señalice que no puede procesar mensajes más rápido. En este proyecto se implementa con el circuit breaker (descarta mensajes cuando la API está caída) y con `INFERENCE_RATE_LIMIT_MS` (pausa configurable entre requests).

**Commit selectivo (Kafka)**
Estrategia de confirmar el offset de Kafka solo cuando el mensaje fue procesado exitosamente de extremo a extremo. Si la inferencia o la publicación falla, el offset no se confirma y el mensaje se reintenta en la siguiente iteración del loop.

**Severidad de alerta**
Clasificación del riesgo de fraude según el score del modelo: `WARNING` (score ≥ 0.50), `HIGH` (score ≥ 0.75), `CRITICAL` (score ≥ 0.90). Determina la urgencia con que el equipo operativo debe responder.

---

## MLOps y ciclo de vida del modelo

**MLflow**
Plataforma para rastrear experimentos de machine learning. Guarda métricas, parámetros, artefactos (gráficos, modelos) y versiones de cada entrenamiento. Accesible en `http://localhost:5000`.

**Model Registry**
Componente de MLflow que gestiona versiones de modelos y sus etapas (`Staging` → `Production`).

**Artifact store**
Almacén de archivos de MLflow donde se guardan los artefactos de cada run: el modelo serializado, los encoders, gráficos de evaluación, el dataset de referencia para drift, y los reportes HTML de Evidently. En este proyecto usa el filesystem del contenedor mlflow montado como volumen Docker.

**Staging**
Etapa inicial de un modelo registrado en MLflow. El modelo existe pero aún no está en producción; primero debe pasar los quality gates.

**Production**
Etapa de un modelo que está activo en producción. Solo puede haber un modelo en `Production` a la vez.

**Archived**
Etapa final de un modelo en MLflow. Se asigna automáticamente cuando un modelo falla los quality gates o es desplazado por un nuevo champion. Los modelos archivados quedan almacenados pero ya no se sirven.

**Quality Gates**
Umbrales mínimos que un modelo debe superar antes de poder promoverse a producción: F1-score >= 0.85, AUC-ROC >= 0.90, latencia P99 <= 50ms.

**Champion / Challenger**
Patrón de comparación de modelos. El champion es el modelo actualmente en producción; el challenger es el candidato nuevo. El challenger solo reemplaza al champion si lo supera en F1 por más de 2 puntos porcentuales.

**Airflow**
Orquestador de workflows. Permite programar y monitorear pipelines de datos y ML. Accesible en `http://localhost:8081`.

**DAG (Directed Acyclic Graph)**
La unidad básica de Airflow: un grafo de tareas con dependencias entre ellas. "Acíclico" significa que nunca hay ciclos; las tareas siempre fluyen hacia adelante. Cada DAG tiene un schedule (cuándo corre) y un conjunto de tasks (qué hace).

**Task (Airflow)**
Unidad mínima de trabajo dentro de un DAG. Cada task hace una cosa: extraer datos, entrenar un modelo, evaluar métricas, etc. Las dependencias entre tasks definen el orden de ejecución.

**Operador (Airflow)**
Clase base de Airflow que implementa la lógica de una task. En este proyecto hay tres operadores custom: `TimescaleExtractOperator` (extrae datos a Parquet), `EvidentlyReportOperator` (ejecuta análisis de drift) y `MLflowRegisterModelOperator` (transiciona versiones en el Registry).

**XCom**
Mecanismo de Airflow para compartir valores entre tasks de un mismo DAG. Una task produce un valor (`return` o `xcom_push`) y otra lo lee (`xcom_pull`). En este proyecto se usa para pasar paths de Parquet, `model_version` y resultados de drift entre tasks.

**Event-driven DAG**
DAG con `schedule=None` que solo corre cuando es disparado explícitamente, ya sea desde otro DAG via `trigger_dag()` o desde la API REST de Airflow. En este proyecto `validate_and_promote_model` es event-driven: solo corre cuando `retrain_fraud_model` termina exitosamente.

**TriggerRule**
Condición bajo la cual Airflow ejecuta una task, más allá de la dependencia estándar (esperan a que las anteriores terminen con éxito). `TriggerRule.ONE_FAILED` ejecuta la task si al menos una dependencia falló o fue skipeada. En este proyecto se usa para archivar versiones rechazadas automáticamente.

**AirflowSkipException**
Excepción especial de Airflow que marca una task como "skipeada" en lugar de "fallida". Se usa cuando la condición de negocio no se cumple (por ejemplo, datos insuficientes para reentrenar) pero no es un error; el DAG simplemente no tiene trabajo que hacer.

**Reentrenamiento automático**
Proceso por el cual el pipeline dispara un nuevo ciclo de entrenamiento sin intervención humana. En este proyecto puede iniciarse de dos formas: por el schedule diario de `retrain_fraud_model` o por el DAG de drift cuando detecta severidad `HIGH` o `CRITICAL`.

**Deployment activo**
El registro en `public.model_deployments` donde `is_active = TRUE`. Identifica la versión del modelo que sirve predicciones en ese momento. Solo puede haber uno activo a la vez; el stored procedure `activate_model_version` garantiza esa invariante con una transacción atómica.

**Evidently AI**
Librería Python para monitoreo de modelos ML. Calcula automáticamente métricas de drift sobre features y performance del modelo, y genera reportes en distintos formatos (dict, HTML). En este proyecto se usa con `DataDriftPreset` para comparar la distribución actual de features contra el dataset de referencia de entrenamiento.

**Data drift**
Cambio en la distribución estadística de las features de entrada con respecto al dataset de referencia. Se detecta feature por feature usando tests estadísticos (Wasserstein para numéricas, chi-cuadrado para categóricas). Si el `drift_share` supera el `DRIFT_THRESHOLD_GLOBAL` (30%), Evidently reporta dataset drift.

**Model drift**
Degradación de las métricas de performance del modelo en producción respecto al baseline de entrenamiento. Se calcula comparando el F1 actual (sobre predicciones con `actual_label` conocido) contra el F1 registrado en `model_deployments`. Si la caída supera 0.05 puntos, se declara model drift.

**Drift score**
Valor numérico entre 0 y 1 que representa la intensidad del data drift en una feature. Un drift score alto indica que la distribución actual se alejó significativamente de la referencia. En el contexto del dataset completo, `drift_share` es la proporción de features con drift detectado.

**Dataset de referencia (reference dataset)**
El conjunto de datos de entrenamiento guardado como artefacto en MLflow (`reference_dataset.parquet`). Sirve como línea base para el análisis de drift: cada vez que corre el DAG de detección, compara los datos de producción recientes contra este archivo, garantizando consistencia entre el modelo activo y su referencia de drift.

**Feature crítica**
Feature que tiene mayor impacto operativo en la detección de fraude y por eso recibe umbrales de drift más estrictos. En este proyecto son cinco: `tx_count_1h`, `amount_sum_1h`, `amount_ratio_vs_user_avg`, `is_country_new`, `seconds_since_last_tx`. Si alguna de ellas deriva, la severidad de la alerta sube directamente a `HIGH` o `CRITICAL`.

**Severidad de drift**
Clasificación de la urgencia de una alerta de drift: `INFO` (dentro de límites aceptables), `WARNING` (drift global superó el umbral pero sin features críticas), `HIGH` (features críticas con drift o degradación del modelo), `CRITICAL` (ambas condiciones al mismo tiempo). Solo `HIGH` y `CRITICAL` disparan reentrenamiento automático.

**Artefacto (MLflow artifact)**
Archivo generado durante un experimento y guardado en MLflow: gráficos de curvas ROC, matriz de confusión, el archivo del modelo entrenado, encoders, dataset de referencia para drift, y reportes HTML de Evidently. Permite reproducir y auditar cualquier run histórico.

**Latencia P99**
El tiempo de respuesta que el 99% de las predicciones no supera. Si P99 = 12ms significa que 99 de cada 100 predicciones tardan menos de 12ms. Es el indicador de rendimiento más relevante en sistemas de tiempo real.

---

## Métricas del modelo

**F1-score**
Métrica que balancea precisión y recall. Va de 0 a 1; más alto es mejor. Es la métrica principal para evaluar el modelo de fraude porque el dataset está desbalanceado.

**Precisión (Precision)**
De todas las transacciones que el modelo marcó como fraude, ¿cuántas realmente lo eran? Alta precisión = pocos falsos positivos (clientes legítimos bloqueados).

**Recall (Sensibilidad)**
De todos los fraudes reales, ¿cuántos detectó el modelo? Alto recall = pocos fraudes sin detectar.

**AUC-ROC**
Métrica que mide la capacidad discriminativa del modelo independientemente del threshold. Va de 0 a 1; 0.9+ es considerado muy bueno.

**PR-AUC**
Área bajo la curva Precisión-Recall. Más informativa que AUC-ROC cuando el dataset está muy desbalanceado (como en fraude).

**Falso positivo (FP)**
Transacción legítima que el modelo marcó como fraude. Impacto: cliente molesto, transacción bloqueada innecesariamente.

**Falso negativo (FN)**
Transacción fraudulenta que el modelo no detectó. Impacto: pérdida económica real. En este proyecto tiene un costo 20x mayor que un FP.

---

## Convenciones del proyecto

**Conventional Commits**
Formato estándar para mensajes de commit: `tipo(scope): descripción`. Tipos usados: `feat` (nueva funcionalidad), `fix` (corrección de bug), `refactor` (reorganización de código), `docs` (documentación), `chore` (tareas de mantenimiento).

**Seed**
Datos sintéticos generados para poblar la base de datos en entornos de desarrollo y pruebas. En este proyecto simula transacciones históricas con patrones de fraude realistas.
