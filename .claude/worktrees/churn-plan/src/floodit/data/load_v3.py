"""Load the v2 dataset joined with session-level v3 features.

Same train/test rows as ``load_v2`` (joined by ``user_pseudo_id``). Sessions are
derived by time-gap sessionization in BigQuery
(``data/queries/flood_it_features_v3_sessions.sql``) because this 2018 export has
no ``ga_session_id``. Imputes "never returned" / single-event nulls and adds two
derived session features.
"""
import pandas as pd

from floodit import config
from floodit.data.load import sha256_of
from floodit.data.load_v2 import load_v2


def _features_v3(verify: bool = True) -> pd.DataFrame:
    path = config.DATA_DIR / "users_features_v3_sessions.csv"
    if verify and sha256_of(path) != config.DATA_HASHES["features_v3"]:
        raise ValueError("features_v3 hash mismatch")
    df = pd.read_csv(path)
    df["sess_min_to_second_session"] = df["sess_min_to_second_session"].fillna(config.NEVER_SENTINEL)
    df["sess_mean_intra_gap_s"] = df["sess_mean_intra_gap_s"].fillna(0)  # single-event users
    return df


def load_v3(name: str, verify: bool = True) -> pd.DataFrame:
    base = load_v2(name, verify=verify)
    feats = _features_v3(verify=verify)
    df = base.merge(feats, on="user_pseudo_id", how="left", validate="one_to_one")
    if df[config.V3_RAW_SESSION].isna().any().any():
        raise ValueError(f"unmatched rows after v3 join for split {name!r}")

    df["sess_returned"] = (df["sess_num_sessions"] >= 2).astype(int)
    df["sess_engagement_early_frac"] = (
        df["sess_engagement_first_1h_sec"] / (df["sess_total_engagement_sec"] + 1)
    )
    return df
