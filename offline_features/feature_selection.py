import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from offline_features.imbalance_strategies import compute_scale_pos_weight

logger = logging.getLogger(__name__)


def compute_xgboost_importance(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> pd.DataFrame:
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
    return X.corr(method="pearson")


def find_redundant_features(X: pd.DataFrame, correlation_threshold: float = 0.85) -> list[tuple[str, str, float]]:
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


def run_boruta(X: pd.DataFrame, y: pd.Series, max_iter: int = 50, random_state: int = 42) -> dict[str, list[str]]:
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


@dataclass
class FeatureSelectionReport:
    all_features: list[str]
    selected_features: list[str]
    dropped_features: list[str]
    drop_reason: dict[str, str]
    importance_df: pd.DataFrame
    redundant_pairs: list[tuple[str, str, float]]
    boruta_results: dict[str, list[str]] | None = field(default=None)


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    importance_threshold: float = 0.01,
    correlation_threshold: float = 0.85,
    use_boruta: bool = False,
    random_state: int = 42,
    max_iter: int = 50,
) -> FeatureSelectionReport:
    all_features = X.columns.tolist()

    # Step 1 — XGBoost importances
    logger.info("Computing XGBoost feature importances...")
    importance_df = compute_xgboost_importance(X, y, random_state=random_state)
    importance_map: dict[str, float] = dict(zip(importance_df["feature"], importance_df["importance"], strict=False))

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
        logger.info("Running Boruta (max_iter=%d)...", max_iter)
        boruta_results = run_boruta(X, y, random_state=random_state)
        for feat in boruta_results["rejected"]:
            if feat not in drop_reason:
                drop_reason[feat] = "boruta_rejected"

    # Build final lists preserving original column order
    dropped_features = [f for f in all_features if f in drop_reason]
    selected_features = [f for f in all_features if f not in drop_reason]

    logger.info("Feature selection complete: %d selected, %d dropped", len(selected_features), len(dropped_features))

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
