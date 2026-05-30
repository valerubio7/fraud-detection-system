"""
Lista definitiva de features para el FraudDetectionModel.
Auto-actualizada por setup.sh tras la seleccion de features sobre el dataset de entrenamiento.
"""

SELECTED_FEATURES: list[str] = [
    "log_amount",
    "hour_of_day",
    "day_of_week",
    "merchant_category_encoded",
    "country_encoded",
    "device_type_encoded",
    "tx_count_1h",
    "tx_count_24h",
    "tx_count_7d",
    "amount_sum_1h",
    "amount_sum_24h",
    "seconds_since_last_tx",
    "amount_ratio_vs_user_avg",
    "is_country_new",
    "is_merchant_new",
    "distinct_merchants_seen",
]
