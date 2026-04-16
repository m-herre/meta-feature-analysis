from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class AnalysisUnit(str, Enum):
    DATASET = "dataset"
    FOLD = "fold"


class CorrelationMethod(str, Enum):
    SPEARMAN = "spearman"
    PEARSON = "pearson"


class FDRMethod(str, Enum):
    BH = "bh"
    HOLM = "holm"


class MultivariateMethod(str, Enum):
    OLS = "ols"
    RIDGE = "ridge"


@dataclass(frozen=True)
class GroupDef:
    name: str
    config_types: frozenset[str]
    label: str


@dataclass(frozen=True)
class ComparisonSpec:
    name: str
    group_a: GroupDef
    group_b: GroupDef
    expected_direction: str | None = None


@dataclass(frozen=True)
class CorrelationResult:
    comparison_name: str
    predictor: str
    target: str
    statistic: float
    p_value: float
    n_observations: int
    ci_lower: float | None = None
    ci_upper: float | None = None
    direction_confirmed: bool | None = None


@dataclass(frozen=True)
class CorrectionResult:
    method: str
    alpha: float
    results: tuple[CorrelationResult, ...]
    adjusted_p_values: tuple[float, ...]
    rejected: tuple[bool, ...]


@dataclass(frozen=True)
class MultivariateResult:
    comparison_name: str
    predictors: tuple[str, ...]
    coefficients: dict[str, float]
    p_values: dict[str, float]
    r_squared: float
    adj_r_squared: float
    vif: dict[str, float]
    n_observations: int


@dataclass
class AnalysisResult:
    config_hash: str
    comparison_name: str | None
    correlation_results: list[CorrelationResult]
    correction_result: CorrectionResult | None
    multivariate_result: MultivariateResult | None
    gap_table: pd.DataFrame
    metafeature_table: pd.DataFrame
    analysis_table: pd.DataFrame

