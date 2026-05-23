from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class TargetEncoder:
    def __init__(self, smoothing: float = 10.0) -> None:
        self.smoothing = smoothing
        self.mapping_: dict[str, float] = {}
        self.global_mean_: float = 0.0

    def fit(self, X: pd.Series, y: pd.Series) -> TargetEncoder:
        self.global_mean_ = float(y.mean())
        stats = pd.DataFrame({"X": X.astype(str).values, "y": y.values}).groupby("X")["y"].agg(["sum", "count"])
        for category, row in stats.iterrows():
            n = int(row["count"])
            cat_mean = float(row["sum"]) / n
            smoothed = (n * cat_mean + self.smoothing * self.global_mean_) / (n + self.smoothing)
            self.mapping_[str(category)] = smoothed
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        return X.astype(str).map(self.mapping_).fillna(self.global_mean_)

    def fit_transform(self, X: pd.Series, y: pd.Series) -> pd.Series:
        return self.fit(X, y).transform(X)


class OrdinalEncoder:
    def __init__(self) -> None:
        self.mapping_: dict[str, int] = {}

    def fit(self, X: pd.Series) -> OrdinalEncoder:
        categories = X.astype(str).unique()  # preserves order of first appearance
        self.mapping_ = {cat: i for i, cat in enumerate(categories)}
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        return X.astype(str).map(self.mapping_).fillna(-1).astype(np.int64)

    def fit_transform(self, X: pd.Series) -> pd.Series:
        return self.fit(X).transform(X)


class CategoricalEncoderPipeline:
    def __init__(self, smoothing: float = 10.0) -> None:
        self._smoothing = smoothing
        self._merchant_category_enc: TargetEncoder = TargetEncoder(smoothing=smoothing)
        self._country_enc: TargetEncoder = TargetEncoder(smoothing=smoothing)
        self._device_type_enc: OrdinalEncoder = OrdinalEncoder()

    def fit(self, df: pd.DataFrame, y: pd.Series) -> CategoricalEncoderPipeline:
        self._merchant_category_enc.fit(df["merchant_category"], y)
        self._country_enc.fit(df["country"], y)
        self._device_type_enc.fit(df["device_type"])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["merchant_category_encoded"] = self._merchant_category_enc.transform(out["merchant_category"])
        out["country_encoded"] = self._country_enc.transform(out["country"])
        out["device_type_encoded"] = self._device_type_enc.transform(out["device_type"])
        return out

    def save(self, path: str | Path) -> None:
        joblib.dump(self, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> CategoricalEncoderPipeline:
        return joblib.load(Path(path))


__all__ = ["TargetEncoder", "OrdinalEncoder", "CategoricalEncoderPipeline"]
