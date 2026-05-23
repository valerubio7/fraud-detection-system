# ruff: noqa: E402
import sys
from unittest.mock import MagicMock

# Los encoders necesitan pandas y joblib reales. El conftest raíz los stubbea con
# MagicMock si no están instalados. Aquí solo limpiamos stubs, nunca módulos reales.
for _stub in ["pandas", "joblib"]:
    _mod = sys.modules.get(_stub)
    if isinstance(_mod, MagicMock):
        sys.modules.pop(_stub, None)
        for _key in list(sys.modules):
            if _key.startswith(_stub + "."):
                sys.modules.pop(_key, None)

import pandas as pd
import pytest

from offline_features.encoders import CategoricalEncoderPipeline, OrdinalEncoder, TargetEncoder


class TestTargetEncoder:
    def _fit(self, categories, labels):
        enc = TargetEncoder(smoothing=0.0)
        enc.fit(pd.Series(categories), pd.Series(labels))
        return enc

    def test_known_category_gets_its_mean(self):
        enc = self._fit(["A", "A", "B"], [1, 1, 0])
        result = enc.transform(pd.Series(["A"]))
        assert result.iloc[0] == pytest.approx(1.0)

    def test_unknown_category_falls_back_to_global_mean(self):
        enc = self._fit(["A", "B"], [1, 0])
        result = enc.transform(pd.Series(["X"]))
        assert result.iloc[0] == pytest.approx(0.5)

    def test_smoothing_pulls_towards_global_mean(self):
        # "A" has cat_mean=1.0 (1 sample), "B" has label 0.0 → global_mean=0.5
        # smoothed_A = (1*1.0 + 10*0.5) / (1+10) = 6/11 ≈ 0.545
        enc = TargetEncoder(smoothing=10.0)
        enc.fit(pd.Series(["A", "B"]), pd.Series([1.0, 0.0]))
        result = enc.transform(pd.Series(["A"])).iloc[0]
        assert result < 1.0
        assert result > 0.0

    def test_fit_transform_is_equivalent_to_fit_then_transform(self):
        X = pd.Series(["A", "B", "A", "C"])
        y = pd.Series([1, 0, 1, 0])
        enc = TargetEncoder(smoothing=0.0)
        result_ft = enc.fit_transform(X, y)
        result_t = enc.transform(X)
        pd.testing.assert_series_equal(result_ft, result_t)


class TestOrdinalEncoder:
    def test_assigns_sequential_integers(self):
        enc = OrdinalEncoder()
        enc.fit(pd.Series(["mobile", "desktop", "tablet"]))
        result = enc.transform(pd.Series(["mobile", "desktop", "tablet"]))
        assert set(result.values) == {0, 1, 2}

    def test_unknown_category_gets_minus_one(self):
        enc = OrdinalEncoder()
        enc.fit(pd.Series(["mobile", "desktop"]))
        result = enc.transform(pd.Series(["kiosk"]))
        assert result.iloc[0] == -1

    def test_fit_transform_is_equivalent(self):
        X = pd.Series(["a", "b", "c", "a"])
        enc = OrdinalEncoder()
        result_ft = enc.fit_transform(X)
        result_t = enc.transform(X)
        pd.testing.assert_series_equal(result_ft, result_t)

    def test_preserves_first_appearance_order(self):
        enc = OrdinalEncoder()
        enc.fit(pd.Series(["z", "a", "m"]))
        assert enc.mapping_["z"] == 0
        assert enc.mapping_["a"] == 1
        assert enc.mapping_["m"] == 2


class TestCategoricalEncoderPipeline:
    @pytest.fixture
    def fitted_pipeline(self):
        df = pd.DataFrame(
            {
                "merchant_category": ["grocery", "grocery", "electronics"],
                "country": ["AR", "BR", "AR"],
                "device_type": ["mobile", "desktop", "mobile"],
            }
        )
        y = pd.Series([0, 1, 0])
        pipeline = CategoricalEncoderPipeline(smoothing=0.0)
        pipeline.fit(df, y)
        return pipeline

    def test_transform_adds_encoded_columns(self, fitted_pipeline):
        df = pd.DataFrame(
            {
                "merchant_category": ["grocery"],
                "country": ["AR"],
                "device_type": ["mobile"],
            }
        )
        result = fitted_pipeline.transform(df)
        assert "merchant_category_encoded" in result.columns
        assert "country_encoded" in result.columns
        assert "device_type_encoded" in result.columns

    def test_transform_output_is_numeric(self, fitted_pipeline):
        df = pd.DataFrame(
            {
                "merchant_category": ["grocery"],
                "country": ["AR"],
                "device_type": ["mobile"],
            }
        )
        result = fitted_pipeline.transform(df)
        assert pd.api.types.is_numeric_dtype(result["merchant_category_encoded"])
        assert pd.api.types.is_numeric_dtype(result["country_encoded"])
        assert pd.api.types.is_numeric_dtype(result["device_type_encoded"])

    def test_save_and_load_roundtrip(self, fitted_pipeline, tmp_path):
        path = tmp_path / "encoder.joblib"
        fitted_pipeline.save(path)
        loaded = CategoricalEncoderPipeline.load(path)
        df = pd.DataFrame(
            {
                "merchant_category": ["grocery"],
                "country": ["AR"],
                "device_type": ["mobile"],
            }
        )
        result_original = fitted_pipeline.transform(df)
        result_loaded = loaded.transform(df)
        pd.testing.assert_frame_equal(result_original, result_loaded)
