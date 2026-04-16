from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd

from mfa.metafeatures.pymfe_features import extract_pymfe_features


def test_extract_pymfe_features_imputes_numeric_missing_values_with_median(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeMFE:
        def __init__(self, *, groups, summary) -> None:
            captured["groups"] = groups
            captured["summary"] = summary

        def fit(self, X, y, cat_cols) -> None:
            captured["X"] = X.copy()
            captured["y"] = y.copy()
            captured["cat_cols"] = list(cat_cols)

        def extract(self):
            return ["dummy"], [1.0]

    pymfe_module = types.ModuleType("pymfe")
    pymfe_module.__path__ = []
    mfe_module = types.ModuleType("pymfe.mfe")
    mfe_module.MFE = FakeMFE
    pymfe_module.mfe = mfe_module
    monkeypatch.setitem(sys.modules, "pymfe", pymfe_module)
    monkeypatch.setitem(sys.modules, "pymfe.mfe", mfe_module)

    X_train = pd.DataFrame(
        {
            "num": [1.0, np.nan, 5.0, 9.0],
            "cat": pd.Series(["a", "a", None, "b"], dtype="category"),
        }
    )
    y_train = pd.Series([0, 1, 0, 1])

    features = extract_pymfe_features(X_train, y_train, groups=("general",), summary=("mean",))

    np.testing.assert_allclose(captured["X"], np.array([[1.0, 0.0], [5.0, 0.0], [5.0, 0.0], [9.0, 1.0]]))
    np.testing.assert_array_equal(captured["y"], y_train.to_numpy())
    assert captured["cat_cols"] == [1]
    assert captured["groups"] == ["general"]
    assert captured["summary"] == ["mean"]
    assert features == {"pymfe__dummy": 1.0}
