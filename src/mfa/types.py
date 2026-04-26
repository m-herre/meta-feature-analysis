from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd


class AnalysisUnit(StrEnum):
    DATASET = "dataset"
    FOLD = "fold"


class CorrelationMethod(StrEnum):
    SPEARMAN = "spearman"
    PEARSON = "pearson"


class FDRMethod(StrEnum):
    BH = "bh"
    HOLM = "holm"


class MultivariateMethod(StrEnum):
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


@dataclass
class AnalysisResult:
    config_hash: str
    comparison_name: str | None
    gap_table: pd.DataFrame
    metafeature_table: pd.DataFrame
    analysis_table: pd.DataFrame
    correlation_results: list = field(default_factory=list)
    correction_result: object | None = None
    multivariate_result: object | None = None
