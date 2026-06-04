"""Load the v3 dataset joined with navigation-sequence v4 features.

Same train/test rows as ``load_v3``. Navigation features come from
``data/queries/flood_it_features_v4_navigation.sql``, which maps platform-
specific screen classes to logical categories and counts visits/transitions
within the 24 h window. Users with no screen views in window get all-zero nav
features. Adds a few derived navigation ratios.
"""
import pandas as pd

from floodit import config
from floodit.data.load import sha256_of
from floodit.data.load_v3 import load_v3


def _features_v4(verify: bool = True) -> pd.DataFrame:
    path = config.DATA_DIR / "users_features_v4_navigation.csv"
    if verify and sha256_of(path) != config.DATA_HASHES["features_v4"]:
        raise ValueError("features_v4 hash mismatch")
    return pd.read_csv(path)


def load_v4(name: str, verify: bool = True) -> pd.DataFrame:
    base = load_v3(name, verify=verify)
    feats = _features_v4(verify=verify)
    df = base.merge(feats, on="user_pseudo_id", how="left", validate="one_to_one")
    # Users with no screen views in window -> no navigation.
    df[config.V4_RAW_NAV] = df[config.V4_RAW_NAV].fillna(0)

    df["nav_transition_rate"] = df["nav_transitions"] / (df["nav_screen_views"] + 1)
    df["nav_revisit_rate"] = df["nav_self_loops"] / (df["nav_transitions"] + 1)
    df["nav_game_over_rate"] = df["nav_cnt_game_over"] / (df["nav_screen_views"] + 1)
    df["nav_reached_shop"] = (df["nav_cnt_shop"] > 0).astype(int)
    df["nav_reached_steps"] = (df["nav_cnt_steps"] > 0).astype(int)
    return df
