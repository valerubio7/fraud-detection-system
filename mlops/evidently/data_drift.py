from dataclasses import dataclass, field

import pandas as pd


@dataclass
class FeatureDriftResult:
    feature_name: str
    drift_detected: bool
    drift_score: float
    stattest_name: str


@dataclass
class DataDriftResult:
    dataset_drift: bool
    drift_share: float
    drifted_features: list[str]
    feature_results: dict[str, FeatureDriftResult] = field(default_factory=dict)


def _build_and_run_report(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, columns: list[str]
) -> tuple[DataDriftResult, object]:
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df[columns], current_data=current_df[columns])
    raw = report.as_dict()

    metrics = raw.get("metrics", [])

    dataset_metric = next((m for m in metrics if "share_of_drifted_columns" in m.get("result", {})), {})
    dataset_result = dataset_metric.get("result", {})
    dataset_drift: bool = bool(dataset_result.get("dataset_drift", False))
    drift_share: float = float(dataset_result.get("share_of_drifted_columns", 0.0))

    column_metric = next((m for m in metrics if "drift_by_columns" in m.get("result", {})), {})
    drift_by_columns: dict = column_metric.get("result", {}).get("drift_by_columns", {})

    feature_results: dict[str, FeatureDriftResult] = {}
    for col in columns:
        col_data = drift_by_columns.get(col, {})
        feature_results[col] = FeatureDriftResult(
            feature_name=col,
            drift_detected=bool(col_data.get("drift_detected", False)),
            drift_score=float(col_data.get("drift_score", col_data.get("stattest_threshold", 0.0))),
            stattest_name=str(col_data.get("stattest_name", "")),
        )

    drifted_features = [name for name, fr in feature_results.items() if fr.drift_detected]

    result = DataDriftResult(
        dataset_drift=dataset_drift,
        drift_share=drift_share,
        drifted_features=drifted_features,
        feature_results=feature_results,
    )
    return result, report


def run_data_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame, columns: list[str]) -> DataDriftResult:
    result, _ = _build_and_run_report(reference_df, current_df, columns)
    return result


def run_data_drift_report_with_html(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    columns: list[str],
    html_path: str = "/tmp/data_drift_report.html",
) -> tuple[DataDriftResult, str]:
    result, report = _build_and_run_report(reference_df, current_df, columns)
    report.save_html(html_path)
    return result, html_path
