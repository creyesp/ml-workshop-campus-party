"""Validation strategy (spec §6), committed in code before any model is fit.

Why stratified k-fold is the primary strategy: the unit of analysis is one
row per ``user_pseudo_id`` (spec §3), so rows are independent — there is no
group structure forcing a grouped split, and stratifying on ``churned`` keeps
the ~23% prevalence stable across folds, which matters for a minority-class
PR-AUC estimate on a small dataset.

Why a time-based split is also provided (robustness check, spec §6/§8): the
underlying process is cohorts arriving over calendar time
(``user_first_engagement``); if it is non-stationary, a random split would be
optimistic. ``time_based_split`` trains on earlier cohorts and validates on
later ones to surface that.
"""
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from floodit import config


def stratified_kfold(
    df: pd.DataFrame, y, n_splits: int = 5
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) positional-index arrays, stratified on ``y``."""
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED
    )
    X_placeholder = np.zeros(len(df))
    yield from skf.split(X_placeholder, y)


def time_based_split(
    df: pd.DataFrame, frac: float = 0.8
) -> tuple[np.ndarray, np.ndarray]:
    """Earlier ``frac`` of cohorts -> train, the rest -> validation.

    Returns positional indices into ``df`` (not label values), so the caller
    can apply them with ``df.iloc[...]``.
    """
    order = np.argsort(df["user_first_engagement"].values, kind="stable")
    cut = int(round(frac * len(order)))
    return order[:cut], order[cut:]
