# Hallazgos del análisis exploratorio de datos (EDA)

## Resumen ejecutivo

El dataset analizado contiene 10 000 transacciones bancarias sintéticas generadas con seed fijo (seed=42), distribuidas en una ventana de 30 días. Cada transacción incluye información del usuario, comercio, monto, país, dispositivo y marca temporal. La tasa de fraude es del **2 %** (200 transacciones fraudulentas sobre 9 800 legítimas). Los patrones de fraude fueron inyectados de forma controlada siguiendo cuatro modelos: monto atípico, país inusual, alta frecuencia y merchant desconocido con monto elevado. El objetivo de este análisis es identificar las features con mayor poder discriminativo y definir las decisiones de ingeniería de features.

---

## Distribución de clases

El dataset presenta un **desbalance severo**: el 98 % de las transacciones son legítimas y solo el 2 % son fraudulentas. Este nivel de desbalance implica que un clasificador trivial que prediga siempre "legítimo" alcanzaría 98 % de accuracy sin aprender ningún patrón. Para el entrenamiento del modelo se deberá compensar este desbalance mediante técnicas específicas, como sobremuestreo de la clase minoritaria (SMOTE) o ajuste del peso de la clase positiva (`scale_pos_weight` en XGBoost o `class_weight='balanced'` en scikit-learn). La métrica de evaluación principal no debe ser accuracy sino **precision-recall AUC** o **F1 sobre la clase fraude**.

---

## Hallazgos sobre la distribución de montos

Existe una diferencia clara entre los montos de transacciones fraudulentas y legítimas. El monto medio de las transacciones fraudulentas (~$607) es aproximadamente cuatro veces mayor al de las legítimas (~$143), y la diferencia en la mediana es aún más pronunciada (~$529 vs ~$57). Esto confirma el patrón de *monto atípico* inyectado por el seed (multiplicador 5x–10x sobre el promedio del usuario).

La distribución de montos es fuertemente asimétrica a la derecha en ambas clases. La transformación **`log_amount`** reduce esta asimetría y es candidata a incluirse como feature en lugar del monto bruto. La correlación de Pearson de `amount` con `is_fraud` es 0.24 y la de `log_amount` es 0.18, siendo ambas las más altas del conjunto. La importancia conjunta de ambas en el RandomForest supera el 46 %, lo que confirma que el monto es la señal individual más fuerte.

---

## Hallazgos temporales

El análisis por hora del día muestra que la **hora 06h presenta la mayor tasa de fraude (6.17 %)**, a pesar de tener bajo volumen de transacciones, lo que indica una concentración desproporcionada de fraude en ese horario. El heatmap hora × día de la semana refuerza este patrón: la combinación **lunes a las 06h** alcanza una tasa de fraude del 33 % sobre un volumen reducido.

Por día de la semana, el **lunes muestra la mayor tasa (3.12 %)** y el miércoles la menor (1.07 %), aunque la distribución diaria del seed es uniforme, por lo que estas diferencias deben interpretarse con cautela. Se detectó un día outlier (**2026-04-20** con 7.28 % de tasa), probablemente un artefacto estadístico del seed.

Dado que la distribución horaria no es uniforme y se observan ventanas de mayor riesgo, **`hour_of_day` es una feature candidata relevante**. `day_of_week` tiene menor señal en datos sintéticos pero debería retenerse para producción, donde el comportamiento entre días hábiles y fin de semana suele diferir. Ambas features deberán codificarse con **encoding cíclico (sin/cos)** para evitar la discontinuidad en los extremos (hora 23→0, domingo→lunes).

---

## Hallazgos sobre variables categóricas

- **País (`country`)**: Argentina concentra el 74 % del volumen. La correlación de `country_enc` con `is_fraud` es 0.07, siendo la categórica más informativa. El patrón de *país inusual* inyectado por el seed es capturable si se modela la desviación del país habitual del usuario.
- **Categoría de merchant (`merchant_category`)**: Es la feature con mayor importancia en el RandomForest (~40 %), lo que sugiere que ciertos tipos de comercio están sobrerepresentados en fraude (vinculado al patrón de *merchant desconocido con monto alto*). La distribución entre las 7 categorías no es uniforme en términos de tasa de fraude.
- **Tipo de dispositivo (`device_type`)**: Distribución mobile 60 % / web 30 % / pos 10 %. La importancia en el modelo simple es ~1 %, lo que indica señal débil en los datos actuales.

Para las tres variables se requerirá encoding. El **target encoding** es el candidato preferido dado que preserva la relación con la variable objetivo, pero introduce riesgo de data leakage si no se aplica con cross-validation.

---

## Correlaciones y features redundantes

El análisis de correlación de Pearson sobre las features numéricas construidas no encontró **ningún par con |r| > 0.85**. En particular, `amount` y `log_amount` tienen una correlación alta entre sí pero no superan el umbral definido.

El modelo simple (RandomForestClassifier con `n_estimators=50`, `max_depth=5`) no identificó **ninguna feature con importancia < 1 %**. En consecuencia, no hay features candidatas a eliminación por redundancia o por falta de señal en este conjunto de datos.

| Feature | Importancia (RF) | Correlación con `is_fraud` |
|---|---|---|
| `merchant_category_enc` | 39.9 % | 0.011 |
| `amount` | 24.0 % | 0.239 |
| `log_amount` | 22.5 % | 0.179 |
| `country_enc` | 5.1 % | 0.072 |
| `hour_of_day` | 4.7 % | 0.007 |
| `day_of_week` | 2.8 % | −0.013 |
| `device_type_enc` | 1.15 % | 0.008 |

---

## Features candidatas para el modelo

### Features directas (disponibles en el stream en tiempo real)

| Feature | Justificación |
|---|---|
| `log_amount` | Mayor poder discriminativo de las numéricas; reduce asimetría de la distribución |
| `merchant_category` | Feature más importante en el modelo exploratorio (~40 % importancia RF) |
| `country` | Captura el patrón de país inusual; correlación 0.07 con `is_fraud` |
| `hour_of_day` (sin/cos) | Ventanas horarias de riesgo observables; codificación cíclica necesaria |
| `device_type` | Señal baja en datos sintéticos pero relevante en producción; bajo costo computacional |
| `day_of_week` (sin/cos) | Señal limitada en datos sintéticos; retener para patrones de producción |

### Features derivadas

| Feature | Justificación |
|---|---|
| Frecuencia de transacciones del usuario (ventana 30 min) | Captura el patrón de alta frecuencia del seed |
| Desviación del monto respecto al promedio histórico del usuario | Captura el patrón de monto atípico del seed |
| Indicador de merchant nunca visto por el usuario | Captura el patrón de merchant desconocido del seed |
| Indicador de país distinto al habitual del usuario | Captura el patrón de país inusual del seed |

---

## Decisiones de feature engineering

- **Transformación de monto**: usar `log_amount` como feature principal; descartar `amount` bruto del vector de entrada del modelo para evitar redundancia.
- **Encoding de categóricas**: aplicar *target encoding* con validación cruzada para `merchant_category`, `country` y `device_type`, evitando data leakage.
- **Encoding temporal**: codificar `hour_of_day` y `day_of_week` con sin/cos para preservar la continuidad cíclica.
- **Manejo del desbalance**: evaluar SMOTE en el conjunto de entrenamiento y/o `scale_pos_weight` en XGBoost. La métrica de selección de modelo será F1 sobre la clase fraude y precision-recall AUC.
- **Features a descartar**: `amount` bruto (reemplazado por `log_amount`), `transaction_id`, `user_id`, `merchant_id`, `ip_hash` (identificadores únicos sin poder predictivo directo en el stream).

---

## Limitaciones del análisis

Los datos analizados son **completamente sintéticos** y los patrones de fraude fueron inyectados de forma deliberada y conocida. Esto implica que las correlaciones observadas y las importancias de features reflejan los mecanismos del seed, no necesariamente el comportamiento de fraude real. En producción, los patrones pueden ser más complejos, menos limpios y evolucionar con el tiempo. El análisis temporal también está limitado por la ventana de 30 días y la distribución uniforme por día de semana del seed. Los hallazgos documentados aquí deben tratarse como hipótesis a validar con datos reales antes de tomar decisiones de diseño irreversibles en el pipeline de producción.
