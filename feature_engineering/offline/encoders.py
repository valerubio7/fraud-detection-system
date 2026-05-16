"""
Categorical encoders for fraud detection feature engineering.

Provides TargetEncoder (high-cardinality columns: merchant_category, country),
OrdinalEncoder (low-cardinality columns: device_type), and
CategoricalEncoderPipeline (applies the right encoder to each column).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class TargetEncoder:
    """Smoothed target encoder for categorical variables.

    Replaces each category with a smoothed estimate of the mean target value,
    reducing overfitting on low-frequency categories.

    Smoothing formula::

        encoded = (n * cat_mean + smoothing * global_mean) / (n + smoothing)

    where ``n`` is the number of observations for the category.

    Attributes:
        mapping_: Category → smoothed target mean (populated after fit).
        global_mean_: Global target mean used as fallback for unseen categories.
    """

    def __init__(self, smoothing: float = 10.0) -> None:
        """
        Args:
            smoothing: Regularisation strength. Higher values pull rare-category
                estimates closer to the global mean.
        """
        self.smoothing = smoothing
        self.mapping_: dict[str, float] = {}
        self.global_mean_: float = 0.0

    def fit(self, X: pd.Series, y: pd.Series) -> TargetEncoder:
        """Compute smoothed target means per category.

        Args:
            X: Categorical feature series.
            y: Binary target series (e.g. is_fraud).

        Returns:
            self, for method chaining.
        """
        self.global_mean_ = float(y.mean())
        stats = (
            pd.DataFrame({"X": X.astype(str).values, "y": y.values})
            .groupby("X")["y"]
            .agg(["sum", "count"])
        )
        for category, row in stats.iterrows():
            n = int(row["count"])
            cat_mean = float(row["sum"]) / n
            smoothed = (n * cat_mean + self.smoothing * self.global_mean_) / (n + self.smoothing)
            self.mapping_[str(category)] = smoothed
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        """Replace each category with its smoothed target mean.

        Unseen categories fall back to ``global_mean_``.

        Args:
            X: Categorical feature series (may contain unseen categories).

        Returns:
            Float series of encoded values, indexed identically to X.
        """
        return X.astype(str).map(self.mapping_).fillna(self.global_mean_)

    def fit_transform(self, X: pd.Series, y: pd.Series) -> pd.Series:
        return self.fit(X, y).transform(X)


class OrdinalEncoder:
    """Ordinal encoder for low-cardinality categorical variables.

    Assigns an integer to each category in order of first appearance during fit.
    Unseen categories are encoded as ``-1``.

    Attributes:
        mapping_: Category → integer ordinal (populated after fit).
    """

    def __init__(self) -> None:
        self.mapping_: dict[str, int] = {}

    def fit(self, X: pd.Series) -> OrdinalEncoder:
        """Build the category-to-integer mapping from X.

        Categories are numbered 0, 1, 2, … in order of first appearance.

        Args:
            X: Categorical feature series.

        Returns:
            self, for method chaining.
        """
        categories = X.astype(str).unique()  # preserves order of first appearance
        self.mapping_ = {cat: i for i, cat in enumerate(categories)}
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        """Replace each category with its integer ordinal.

        Args:
            X: Categorical feature series (may contain unseen categories).

        Returns:
            Integer series of encoded values, indexed identically to X.
            Unseen categories are encoded as -1.
        """
        return X.astype(str).map(self.mapping_).fillna(-1).astype(np.int64)

    def fit_transform(self, X: pd.Series) -> pd.Series:
        return self.fit(X).transform(X)


class CategoricalEncoderPipeline:
    """Pipeline that applies the appropriate encoder to each categorical column.

    - :class:`TargetEncoder` → ``merchant_category`` (high cardinality)
    - :class:`TargetEncoder` → ``country`` (high cardinality)
    - :class:`OrdinalEncoder` → ``device_type`` (low cardinality, 3 values)

    Example::

        pipeline = CategoricalEncoderPipeline()
        pipeline.fit(df_train, y_train)
        df_encoded = pipeline.transform(df_test)
        pipeline.save("artifacts/encoders/categorical_encoder.joblib")

        # Later:
        pipeline = CategoricalEncoderPipeline.load(
            "artifacts/encoders/categorical_encoder.joblib"
        )
    """

    def __init__(self, smoothing: float = 10.0) -> None:
        """
        Args:
            smoothing: Smoothing strength forwarded to both TargetEncoders.
        """
        self._smoothing = smoothing
        self._merchant_category_enc: TargetEncoder = TargetEncoder(smoothing=smoothing)
        self._country_enc: TargetEncoder = TargetEncoder(smoothing=smoothing)
        self._device_type_enc: OrdinalEncoder = OrdinalEncoder()

    def fit(self, df: pd.DataFrame, y: pd.Series) -> CategoricalEncoderPipeline:
        """Fit all encoders on the training data.

        Args:
            df: DataFrame containing at least ``merchant_category``, ``country``,
                and ``device_type`` columns.
            y: Binary target series aligned with df rows (is_fraud).

        Returns:
            self, for method chaining.
        """
        self._merchant_category_enc.fit(df["merchant_category"], y)
        self._country_enc.fit(df["country"], y)
        self._device_type_enc.fit(df["device_type"])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted encoders and return a copy with the encoded columns appended.

        Args:
            df: DataFrame containing ``merchant_category``, ``country``, ``device_type``.

        Returns:
            Copy of df with three new float/int columns appended:
            ``merchant_category_encoded``, ``country_encoded``, ``device_type_encoded``.
        """
        out = df.copy()
        out["merchant_category_encoded"] = self._merchant_category_enc.transform(
            out["merchant_category"]
        )
        out["country_encoded"] = self._country_enc.transform(out["country"])
        out["device_type_encoded"] = self._device_type_enc.transform(out["device_type"])
        return out

    def save(self, path: str | Path) -> None:
        """Serialise the fitted pipeline to disk with joblib.

        Args:
            path: Destination file path.
        """
        joblib.dump(self, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> CategoricalEncoderPipeline:
        """Deserialise a previously saved pipeline from disk.

        Args:
            path: Path to the joblib file written by :meth:`save`.

        Returns:
            Fitted CategoricalEncoderPipeline instance.
        """
        return joblib.load(Path(path))


__all__ = ["TargetEncoder", "OrdinalEncoder", "CategoricalEncoderPipeline"]
