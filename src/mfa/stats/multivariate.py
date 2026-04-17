from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ..types import MultivariateMethod, MultivariateResult


def _compute_vif(X: pd.DataFrame) -> dict[str, float]:
    if X.shape[1] <= 1:
        return {column: 1.0 for column in X.columns}
    values = X.to_numpy()
    return {column: float(variance_inflation_factor(values, idx)) for idx, column in enumerate(X.columns)}


def run_multivariate(
    analysis_table: pd.DataFrame,
    *,
    comparison_name: str,
    predictors: Sequence[str],
    target: str = "delta_norm",
    method: MultivariateMethod = MultivariateMethod.OLS,
    ridge_alpha: float = 1.0,
) -> MultivariateResult | None:
    """Run an exploratory multivariate model for one comparison."""
    subset = analysis_table[analysis_table["comparison_name"] == comparison_name][list(predictors) + [target]].dropna()
    if len(subset) < max(3, len(predictors) + 1):
        return None

    X = subset[list(predictors)].astype(float)
    y = subset[target].astype(float)
    vif = _compute_vif(X)

    if method == MultivariateMethod.OLS:
        X_const = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, X_const).fit()
        coefficients = {column: float(model.params[column]) for column in X.columns}
        p_values = {column: float(model.pvalues[column]) for column in X.columns}
        r_squared = float(model.rsquared)
        adj_r_squared = float(model.rsquared_adj)
    else:
        X_values = X.to_numpy()
        y_values = y.to_numpy()
        X_centered = X_values - X_values.mean(axis=0, keepdims=True)
        y_centered = y_values - y_values.mean()
        identity = np.eye(X_centered.shape[1])
        coefficients_array = np.linalg.solve(
            X_centered.T @ X_centered + (ridge_alpha * identity),
            X_centered.T @ y_centered,
        )
        predictions = y_values.mean() + (X_centered @ coefficients_array)
        residual_sum_squares = float(np.sum((y_values - predictions) ** 2))
        total_sum_squares = float(np.sum((y_values - y_values.mean()) ** 2))
        r_squared = float(1 - residual_sum_squares / total_sum_squares) if total_sum_squares > 0 else np.nan
        coefficients = {column: float(value) for column, value in zip(X.columns, coefficients_array, strict=False)}
        p_values = {column: np.nan for column in X.columns}
        adj_r_squared = np.nan

    return MultivariateResult(
        comparison_name=comparison_name,
        predictors=tuple(predictors),
        coefficients=coefficients,
        p_values=p_values,
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
        vif=vif,
        n_observations=int(len(subset)),
    )
