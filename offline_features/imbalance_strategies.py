import logging
from dataclasses import dataclass

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

_XGBOOST_BASE_PARAMS: dict = {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1, "eval_metric": "aucpr"}


def compute_scale_pos_weight(y: pd.Series) -> float:
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        raise ValueError("No positive examples (fraud=1) found in y. Cannot compute scale_pos_weight.")
    n_neg = int((y == 0).sum())
    return n_neg / n_pos


def apply_smote(
    X: pd.DataFrame,
    y: pd.Series,
    sampling_strategy: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    counts_before = y.value_counts().to_dict()
    logger.info(
        "Class distribution before SMOTE — legitimate: %d, fraud: %d", counts_before.get(0, 0), counts_before.get(1, 0)
    )

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)

    X_resampled = pd.DataFrame(X_res, columns=X.columns)
    y_resampled = pd.Series(y_res, name=y.name)

    counts_after = y_resampled.value_counts().to_dict()
    logger.info(
        "Class distribution after SMOTE  — legitimate: %d, fraud: %d", counts_after.get(0, 0), counts_after.get(1, 0)
    )

    return X_resampled, y_resampled


def evaluate_imbalance_strategy(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}

    # --- SMOTE strategy ---
    X_smote, y_smote = apply_smote(X_train, y_train, random_state=random_state)
    model_smote = XGBClassifier(**_XGBOOST_BASE_PARAMS, random_state=random_state)
    model_smote.fit(X_smote, y_smote)
    results["smote"] = _compute_metrics(model_smote, X_val, y_val)
    logger.info("SMOTE strategy metrics: %s", results["smote"])

    # --- scale_pos_weight strategy ---
    spw = compute_scale_pos_weight(y_train)
    model_spw = XGBClassifier(**_XGBOOST_BASE_PARAMS, scale_pos_weight=spw, random_state=random_state)
    model_spw.fit(X_train, y_train)
    results["scale_pos_weight"] = _compute_metrics(model_spw, X_val, y_val)
    logger.info("scale_pos_weight=%.2f strategy metrics: %s", spw, results["scale_pos_weight"])

    return results


def _compute_metrics(model: XGBClassifier, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_proba)),
        "average_precision": float(average_precision_score(y, y_proba)),
    }


@dataclass
class ImbalanceReport:
    strategy_results: dict[str, dict[str, float]]
    recommended_strategy: str
    recommended_scale_pos_weight: float | None
    class_distribution: dict[str, int]


def run_imbalance_analysis(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int = 42,
) -> ImbalanceReport:
    class_distribution = {"legitimate": int((y_train == 0).sum()), "fraud": int((y_train == 1).sum())}
    logger.info(
        "Training class distribution — legitimate: %d, fraud: %d",
        class_distribution["legitimate"],
        class_distribution["fraud"],
    )

    strategy_results = evaluate_imbalance_strategy(X_train, y_train, X_val, y_val, random_state)

    recommended = max(strategy_results, key=lambda s: strategy_results[s]["f1"])
    recommended_spw = compute_scale_pos_weight(y_train) if recommended == "scale_pos_weight" else None

    report = ImbalanceReport(
        strategy_results=strategy_results,
        recommended_strategy=recommended,
        recommended_scale_pos_weight=recommended_spw,
        class_distribution=class_distribution,
    )

    best_metrics = strategy_results[recommended]
    logger.info(
        "Recommended strategy: %s | F1=%.4f | Precision=%.4f | Recall=%.4f | ROC-AUC=%.4f | Avg Precision=%.4f",
        recommended,
        best_metrics["f1"],
        best_metrics["precision"],
        best_metrics["recall"],
        best_metrics["roc_auc"],
        best_metrics["average_precision"],
    )

    return report


__all__ = [
    "apply_smote",
    "compute_scale_pos_weight",
    "evaluate_imbalance_strategy",
    "run_imbalance_analysis",
    "ImbalanceReport",
]
