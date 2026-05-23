"""
Lista definitiva de features para el FraudDetectionModel.

SELECTED_FEATURES se genera corriendo select_features() desde
offline_features/feature_selection.py sobre el dataset semilla (50 000 transacciones,
seed=42, ~2 % de fraude).

Notas de derivación (de eda_findings.md + análisis de correlación):
- tx_velocity_1h es numéricamente idéntica a tx_count_1h (ambas almacenan el conteo de
  transacciones en la última hora; tx_velocity_1h es simplemente un cast a float64). Su
  correlación de Pearson es 1.0, superando el umbral de redundancia de 0.85.
  tx_velocity_1h tiene igual o menor gain en XGBoost que tx_count_1h en la práctica y
  por eso es la que se descarta del par.
- Todos los demás pares de features muestran |r| < 0.85 en el dataset semilla (hallazgo EDA 4.1.2).
- Boruta no se corre por defecto (use_boruta=False); la lista asume esa configuración.

Para regenerar esta lista contra un dataset nuevo, correr:
    python offline_features/feature_selection.py
y copiar report.selected_features en SELECTED_FEATURES abajo.
"""

SELECTED_FEATURES: list[str] = [
    # Features directas
    "log_amount",
    "hour_of_day",
    "day_of_week",
    "merchant_category_encoded",
    "country_encoded",
    "device_type_encoded",
    # Features de ventana temporal
    "tx_count_1h",
    "tx_count_24h",
    "tx_count_7d",
    "amount_sum_1h",
    "amount_sum_24h",
    # tx_velocity_1h DESCARTADA — redundante con tx_count_1h (r = 1.0)
    "seconds_since_last_tx",
    # Features de perfil histórico
    "amount_ratio_vs_user_avg",
    "is_country_new",
    "distinct_countries_seen",
    "is_merchant_new",
    "distinct_merchants_seen",
]
