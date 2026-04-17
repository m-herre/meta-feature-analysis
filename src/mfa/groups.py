from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from ._logging import get_logger
from .types import ComparisonSpec, GroupDef

logger = get_logger(__name__)


def validate_groups_against_data(df_results: pd.DataFrame, groups: Mapping[str, GroupDef]) -> set[str]:
    """Warn about configured config types that are absent in the loaded results."""
    available = {value for value in df_results.get("config_type", pd.Series(dtype=object)).dropna().unique()}
    missing: set[str] = set()
    for group in groups.values():
        absent = sorted(group.config_types - available)
        if absent:
            missing.update(absent)
            logger.warning("Group `%s` contains config_types absent from data: %s", group.name, ", ".join(absent))
    return missing


def comparison_to_dict(comparison: ComparisonSpec) -> dict[str, str | None]:
    return {
        "comparison_name": comparison.name,
        "group_a_name": comparison.group_a.name,
        "group_b_name": comparison.group_b.name,
        "group_a_label": comparison.group_a.label,
        "group_b_label": comparison.group_b.label,
        "expected_direction": comparison.expected_direction,
    }
