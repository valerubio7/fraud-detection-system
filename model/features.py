"""
Definitive feature list for the FraudDetectionModel.

SELECTED_FEATURES is populated by running select_features() from
feature_engineering/offline/selection.py on the seed dataset (10 000 transactions,
seed=42, ~2 % fraud rate).

Derivation notes (from eda_findings.md + correlation analysis):
- tx_velocity_1h is numerically identical to tx_count_1h (both store the count of
  transactions in the last hour; tx_velocity_1h is just cast to float64). Their
  Pearson correlation is 1.0, which exceeds the 0.85 redundancy threshold.
  tx_velocity_1h has equal or lower XGBoost gain than tx_count_1h in practice and
  is therefore the one dropped from the pair.
- All other feature pairs show |r| < 0.85 on the seed dataset (EDA finding 4.1.2).
- No feature shows XGBoost gain < 1 % on the seed dataset (EDA finding 4.1.2),
  so no further features are eliminated by the low-importance filter.
- Boruta is not run by default (use_boruta=False); the list below assumes that setting.

To regenerate this list against a fresh dataset, run:
    python feature_engineering/offline/selection.py
and copy report.selected_features into SELECTED_FEATURES below.
"""

# ---------------------------------------------------------------------------
# Final feature list — update after running selection.py on production data
# ---------------------------------------------------------------------------

SELECTED_FEATURES: list[str] = [
    # Direct features
    "log_amount",
    "hour_of_day",
    "day_of_week",
    "merchant_category_encoded",
    "country_encoded",
    "device_type_encoded",
    # Window features
    "tx_count_1h",
    "tx_count_24h",
    "tx_count_7d",
    "amount_sum_1h",
    "amount_sum_24h",
    # tx_velocity_1h DROPPED — redundant with tx_count_1h (r = 1.0)
    "seconds_since_last_tx",
    # Historical profile features
    "amount_ratio_vs_user_avg",
    "is_country_new",
    "distinct_countries_seen",
    "is_merchant_new",
    "distinct_merchants_seen",
]
