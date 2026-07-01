"""Load the v3 dataset joined with difficulty/progression v6 features.

Same train/test rows as ``load_v3``. Difficulty features come from
``data/queries/flood_it_features_v6_difficulty.sql`` (level fail/retry, max level
reached, score). max_level / score nulls ("no level/score events in window") are
imputed to 0. Adds difficulty ratios.
"""
import pandas as pd

from floodit import config
from floodit.data.load import sha256_of
from floodit.data.load_v3 import load_v3


def _features_v6(verify: bool = True) -> pd.DataFrame:
    path = config.DATA_DIR / "users_features_v6_difficulty.csv"
    if verify and sha256_of(path) != config.DATA_HASHES["features_v6"]:
        raise ValueError("features_v6 hash mismatch")
    df = pd.read_csv(path)
    for col in ("diff_max_level", "diff_max_score", "diff_mean_score"):
        df[col] = df[col].fillna(0)
    return df


def load_v6(name: str, verify: bool = True) -> pd.DataFrame:
    base = load_v3(name, verify=verify)
    feats = _features_v6(verify=verify)
    df = base.merge(feats, on="user_pseudo_id", how="left", validate="one_to_one")
    if df[config.V6_RAW_DIFFICULTY].isna().any().any():
        raise ValueError(f"unmatched rows after v6 join for split {name!r}")

    starts = df["cnt_level_start_quickplay"] + 1
    df["diff_fail_rate"] = df["diff_cnt_fail"] / starts
    df["diff_retry_rate"] = df["diff_cnt_retry"] / starts
    df["diff_fail_to_complete"] = df["diff_cnt_fail"] / (df["cnt_level_complete_quickplay"] + 1)
    return df
