from __future__ import annotations

import pandas as pd


def decode_split_index(split_index: int, n_folds_per_repeat: int = 3) -> tuple[int, int]:
    """Decode TabArena's encoded split id into (repeat, fold-in-repeat)."""
    if n_folds_per_repeat <= 0:
        raise ValueError("`n_folds_per_repeat` must be positive.")
    split_index = int(split_index)
    return split_index // n_folds_per_repeat, split_index % n_folds_per_repeat


def add_split_columns(
    df: pd.DataFrame,
    *,
    fold_col: str = "fold",
    n_folds_per_repeat: int = 3,
    split_id_col: str = "split_id",
    repeat_col: str = "repeat",
    fold_in_repeat_col: str = "fold_in_repeat",
) -> pd.DataFrame:
    """Annotate a result frame with decoded repeat and within-repeat fold columns."""
    decoded = df.copy()
    decoded[split_id_col] = decoded[fold_col].astype(int)
    decoded[repeat_col] = decoded[split_id_col] // n_folds_per_repeat
    decoded[fold_in_repeat_col] = decoded[split_id_col] % n_folds_per_repeat
    return decoded

