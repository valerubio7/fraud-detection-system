"""
Feature selection for fraud detection model training.

Combines three complementary methods:
- XGBoost feature importance (gain-based) to detect low-signal features.
- Pearson correlation to detect redundant feature pairs.
- Boruta (optional) to confirm statistically relevant features.

Use ``select_features`` as the main entry point; it returns a
``FeatureSelectionReport`` that can be passed to
``TransactionFeaturizer.apply_selection`` to permanently filter the transform output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from feature_engineering.offline.imbalance import compute_scale_pos_weight

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual utilities
# ---------------------------------------------------------------------------


def compute_xgboost_importance(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train a lightweight XGBClassifier and return gain-based feature importances.

    Args:
        X: Feature DataFrame (training set).
        y: Binary label series aligned with X (0 = legitimate, 1 = fraud).
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with columns ``feature`` and ``importance``, sorted from highest
        to lowest importance. Importances sum to 1.
    """
    spw = compute_scale_pos_weight(y)
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="aucpr",
        scale_pos_weight=spw,
        random_state=random_state,
    )
    model.fit(X, y)

    importance_df = (
        pd.DataFrame({"feature": X.columns.tolist(), "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return importance_df


def compute_correlation_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Compute the Pearson correlation matrix for all features in X.

    Args:
        X: Feature DataFrame.

    Returns:
        Square DataFrame of Pearson correlations indexed and columned by feature name.
    """
    return X.corr(method="pearson")


def find_redundant_features(
    X: pd.DataFrame,
    correlation_threshold: float = 0.85,
) -> list[tuple[str, str, float]]:
    """Identify feature pairs whose absolute Pearson correlation exceeds the threshold.

    Each pair is reported once (upper triangle only) and the list is sorted by
    descending absolute correlation.

    Args:
        X: Feature DataFrame.
        correlation_threshold: Absolute correlation value above which a pair is
            considered redundant.

    Returns:
        List of ``(feature_a, feature_b, correlation_value)`` tuples, sorted by
        descending absolute correlation.
    """
    corr = compute_correlation_matrix(X)
    features = corr.columns.tolist()
    n = len(features)

    pairs: list[tuple[str, str, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            val = float(corr.iloc[i, j])
            if abs(val) > correlation_threshold:
                pairs.append((features[i], features[j], val))

    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    return pairs


def run_boruta(
    X: pd.DataFrame,
    y: pd.Series,
    max_iter: int = 50,
    random_state: int = 42,
) -> dict[str, list[str]]:
    """Run the Boruta feature selection algorithm using a RandomForestClassifier.

    Boruta creates random shadow features (permuted copies of originals) and
    iteratively confirms or rejects features whose importance consistently beats
    the maximum shadow-feature importance.

    Args:
        X: Feature DataFrame (training set).
        y: Binary label series aligned with X (0 = legitimate, 1 = fraud).
        max_iter: Maximum number of Boruta iterations.
        random_state: Random seed for reproducibility.

    Returns:
        Dict with three keys:
        - ``"confirmed"``: features confirmed as relevant.
        - ``"tentative"``: features neither confirmed nor rejected.
        - ``"rejected"``: features rejected as non-informative.
    """
    from boruta import BorutaPy  # lazy import — optional dependency

    rf = RandomForestClassifier(
        n_estimators="warn",  # overridden by BorutaPy when n_estimators="auto"
        max_depth=5,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    selector = BorutaPy(rf, n_estimators="auto", max_iter=max_iter, random_state=random_state)
    selector.fit(X.values, y.values)

    col = np.array(X.columns.tolist())
    confirmed = col[selector.support_].tolist()
    tentative = col[selector.support_weak_].tolist()
    rejected = col[~selector.support_ & ~selector.support_weak_].tolist()

    result = {"confirmed": confirmed, "tentative": tentative, "rejected": rejected}
    logger.info("Boruta confirmed (%d): %s", len(confirmed), confirmed)
    logger.info("Boruta tentative (%d): %s", len(tentative), tentative)
    logger.info("Boruta rejected  (%d): %s", len(rejected), rejected)
    return result


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class FeatureSelectionReport:
    """Result of the feature selection process.

    Attributes:
        all_features: Complete ordered list of input features.
        selected_features: Features retained after selection.
        dropped_features: Features eliminated.
        drop_reason: Maps each dropped feature to its primary elimination reason:
            ``"low_importance"``, ``"redundant"``, or ``"boruta_rejected"``.
        importance_df: XGBoost gain importances for all input features.
        redundant_pairs: Correlated feature pairs found above the threshold.
        boruta_results: Output of ``run_boruta`` if ``use_boruta=True``, else ``None``.
    """

    all_features: list[str]
    selected_features: list[str]
    dropped_features: list[str]
    drop_reason: dict[str, str]
    importance_df: pd.DataFrame
    redundant_pairs: list[tuple[str, str, float]]
    boruta_results: dict[str, list[str]] | None = field(default=None)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    importance_threshold: float = 0.01,
    correlation_threshold: float = 0.85,
    use_boruta: bool = False,
    random_state: int = 42,
) -> FeatureSelectionReport:
    """Orchestrate the full feature selection pipeline.

    Steps applied in order:
    1. Compute XGBoost gain importances.
    2. Flag features with importance < ``importance_threshold`` as ``"low_importance"``.
    3. Identify redundant pairs with |r| > ``correlation_threshold``; within each
       pair, flag the feature with **lower** XGBoost importance as ``"redundant"``.
    4. If ``use_boruta=True``, run Boruta and flag rejected features as
       ``"boruta_rejected"``.

    When a feature would qualify for multiple drop reasons, the first applicable
    reason in the order above is recorded.

    Args:
        X: Feature DataFrame (training set, already encoded).
        y: Binary label series aligned with X (0 = legitimate, 1 = fraud).
        importance_threshold: Minimum XGBoost importance to retain a feature.
        correlation_threshold: Absolute Pearson correlation above which a pair is
            considered redundant.
        use_boruta: Whether to run the Boruta algorithm as an additional filter.
        random_state: Random seed forwarded to XGBoost and Boruta.

    Returns:
        FeatureSelectionReport with selection details and recommended feature list.
    """
    all_features = X.columns.tolist()

    # Step 1 — XGBoost importances
    logger.info("Computing XGBoost feature importances...")
    importance_df = compute_xgboost_importance(X, y, random_state=random_state)
    importance_map: dict[str, float] = dict(
        zip(importance_df["feature"], importance_df["importance"], strict=False)
    )

    drop_reason: dict[str, str] = {}

    # Step 2 — Low importance
    for feat, imp in importance_map.items():
        if imp < importance_threshold:
            drop_reason[feat] = "low_importance"
            logger.info("Low importance — dropping '%s' (importance=%.4f)", feat, imp)

    # Step 3 — Redundant pairs
    logger.info("Identifying redundant feature pairs (threshold=%.2f)...", correlation_threshold)
    redundant_pairs = find_redundant_features(X, correlation_threshold=correlation_threshold)
    for feat_a, feat_b, corr_val in redundant_pairs:
        imp_a = importance_map.get(feat_a, 0.0)
        imp_b = importance_map.get(feat_b, 0.0)
        weaker = feat_a if imp_a <= imp_b else feat_b
        if weaker not in drop_reason:
            drop_reason[weaker] = "redundant"
            logger.info(
                "Redundant pair ('%s', '%s', r=%.4f) — dropping '%s' (lower importance)",
                feat_a,
                feat_b,
                corr_val,
                weaker,
            )

    # Step 4 — Boruta (optional)
    boruta_results: dict[str, list[str]] | None = None
    if use_boruta:
        logger.info("Running Boruta (max_iter=%d)...", 50)
        boruta_results = run_boruta(X, y, random_state=random_state)
        for feat in boruta_results["rejected"]:
            if feat not in drop_reason:
                drop_reason[feat] = "boruta_rejected"

    # Build final lists preserving original column order
    dropped_features = [f for f in all_features if f in drop_reason]
    selected_features = [f for f in all_features if f not in drop_reason]

    logger.info(
        "Feature selection complete: %d selected, %d dropped",
        len(selected_features),
        len(dropped_features),
    )

    return FeatureSelectionReport(
        all_features=all_features,
        selected_features=selected_features,
        dropped_features=dropped_features,
        drop_reason=drop_reason,
        importance_df=importance_df,
        redundant_pairs=redundant_pairs,
        boruta_results=boruta_results,
    )


__all__ = [
    "compute_xgboost_importance",
    "compute_correlation_matrix",
    "find_redundant_features",
    "run_boruta",
    "select_features",
    "FeatureSelectionReport",
]


# ---------------------------------------------------------------------------
# Standalone example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import psycopg2  # noqa: E402

    import config  # noqa: E402
    from feature_engineering.offline.featurizer import TransactionFeaturizer  # noqa: E402

    settings = config.timescaledb_settings
    conn = psycopg2.connect(
        host=settings.host,
        port=settings.port,
        user=settings.user,
        password=settings.password,
        dbname=settings.db,
    )
    query = """
        SELECT
            transaction_id, user_id, merchant_id, merchant_category,
            amount, country, device_type, ip_hash, timestamp, is_fraud
        FROM public.transactions
        ORDER BY timestamp
    """
    df = pd.read_sql(query, conn)
    conn.close()
    print(f"Cargadas {len(df):,} transacciones desde TimescaleDB.")

    y_full = df["is_fraud"].astype(int)
    featurizer = TransactionFeaturizer()
    X_full = featurizer.fit_transform(df, y_full)

    split = int(len(X_full) * 0.8)
    X_train = X_full.iloc[:split]
    y_train = y_full.iloc[:split]

    report = select_features(X_train, y_train, use_boruta=False)

    print("\n=== FeatureSelectionReport ===")
    print(f"Features seleccionadas ({len(report.selected_features)}): {report.selected_features}")
    print(f"Features eliminadas   ({len(report.dropped_features)}):")
    for feat in report.dropped_features:
        print(f"  {feat:35s} motivo: {report.drop_reason[feat]}")
    print("\nImportancias XGBoost:")
    print(report.importance_df.to_string(index=False))
    if report.redundant_pairs:
        print("\nPares redundantes:")
        for fa, fb, r in report.redundant_pairs:
            print(f"  {fa} <-> {fb}  r={r:.4f}")
