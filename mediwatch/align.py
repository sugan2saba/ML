# mediwatch/align.py
from __future__ import annotations
import pandas as pd
from typing import Tuple, List

def align_like_train(X_train: pd.DataFrame, X_other: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str], bool]:
    """
    Align X_other to have the same columns & order as X_train.

    - Adds any missing columns (filled with 0)
    - Drops any extra columns
    - Reorders to match X_train
    Returns: (X_other_aligned, missing_cols_added, extra_cols_dropped, changed_flag)
    """
    train_cols = list(X_train.columns)
    other_cols = list(X_other.columns)

    missing = [c for c in train_cols if c not in other_cols]
    extra   = [c for c in other_cols if c not in train_cols]

    changed = False

    # Add missing (zero-fill)
    if missing:
        for c in missing:
            X_other[c] = 0
        changed = True

    # Drop extras
    if extra:
        X_other = X_other.drop(columns=extra)
        changed = True

    # Reorder to match train
    if train_cols != list(X_other.columns):
        # Some of the newly added columns may be at the end; enforce exact order
        X_other = X_other[train_cols]
        changed = True

    return X_other, missing, extra, changed
