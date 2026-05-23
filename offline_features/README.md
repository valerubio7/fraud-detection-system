# offline_features

Prepara los datos para entrenar el modelo de detección de fraude. Toma transacciones históricas crudas y produce las features numéricas que XGBoost necesita para aprender a distinguir fraude de transacciones legítimas.

Se usa únicamente en tiempo de entrenamiento. La contraparte en producción es `streaming/features/`, que hace el mismo cálculo pero transacción por transacción en tiempo real.

## Archivos

### `featurizer.py`

El componente principal. Recibe un DataFrame con millones de transacciones históricas y calcula 18 features por fila, garantizando que cada transacción solo ve información del pasado (sin filtrar datos del futuro).

Las 18 features se dividen en tres grupos:

- **Directas** — atributos de la transacción en sí: monto logarítmico, hora del día, día de la semana, encodings de categoría, país y dispositivo.
- **Ventana temporal** — comportamiento reciente del usuario: cuántas transacciones hizo en la última hora, 24h y 7 días, cuánto gastó, cuándo fue la última.
- **Perfil histórico** — desviación respecto al comportamiento pasado del usuario: si el país o merchant es nuevo para él, cuántos países y merchants distintos ha usado, ratio del monto vs. su promedio histórico.

### `encoders.py`

Convierte columnas de texto en números. `merchant_category` y `country` se encodean con la tasa de fraude promedio de cada categoría (suavizada para categorías raras). `device_type` se encodea con un entero ordinal. Los encoders se serializan a disco y se reutilizan en evaluación y serving.

### `feature_selection.py`

Recibe las 18 features y decide cuáles conservar antes de entrenar. Aplica tres filtros en orden: elimina las de baja importancia según XGBoost, elimina las redundantes por correlación de Pearson alta, y opcionalmente corre Boruta. En el dataset de producción quedaron 16 features (se descartaron `device_type_encoded` por baja importancia y `tx_velocity_1h` por ser idéntica a `tx_count_1h`).

### `imbalance_strategies.py`

Compara dos formas de manejar el desbalance típico de fraude (~49 transacciones legítimas por cada fraude). SMOTE genera ejemplos de fraude sintéticos antes de entrenar; `scale_pos_weight` le dice a XGBoost que cada fraude vale 49 veces más en la función de pérdida. Entrena un modelo con cada estrategia, los evalúa en validación y recomienda el que tenga mejor F1. En producción ganó `scale_pos_weight`.
