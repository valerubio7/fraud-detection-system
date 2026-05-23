# ruff: noqa: E402
import sys
from unittest.mock import MagicMock

# compute_scale_pos_weight usa pd.Series; solo limpiamos el stub si está presente.
for _stub in ["pandas"]:
    _mod = sys.modules.get(_stub)
    if isinstance(_mod, MagicMock):
        sys.modules.pop(_stub, None)
        for _key in list(sys.modules):
            if _key.startswith(_stub + "."):
                sys.modules.pop(_key, None)

import pandas as pd
import pytest

from offline_features.imbalance_strategies import compute_scale_pos_weight


class TestComputeScalePosWeight:
    def test_correct_ratio(self):
        y = pd.Series([0] * 90 + [1] * 10)
        result = compute_scale_pos_weight(y)
        assert result == pytest.approx(9.0)

    def test_balanced_dataset_returns_one(self):
        y = pd.Series([0] * 50 + [1] * 50)
        result = compute_scale_pos_weight(y)
        assert result == pytest.approx(1.0)

    def test_raises_when_no_positive_examples(self):
        y = pd.Series([0] * 10)
        with pytest.raises(ValueError, match="No positive examples"):
            compute_scale_pos_weight(y)

    def test_single_fraud_case(self):
        y = pd.Series([0] * 999 + [1])
        result = compute_scale_pos_weight(y)
        assert result == pytest.approx(999.0)
