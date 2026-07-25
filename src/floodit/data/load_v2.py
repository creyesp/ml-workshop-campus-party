"""Load the existing split joined with engineered v2 features.

Keeps the SAME train/test rows and labels as ``load_dataset`` (joined by
``user_pseudo_id``) so a v1-vs-v2 comparison isolates the feature effect. Adds
the BigQuery-derived v2 features (``data/queries/flood_it_features_v2.sql``),
imputes "never did it in window" times to a post-window sentinel, drops the
all-zero ``num_sessions``, and computes a few ratio features.
"""
import pandas as pd

from floodit import config
from floodit.data.load import dataset_path, load_dataset, sha256_of


def _features_v2(verify: bool = True) -> pd.DataFrame:
    path = config.DATA_DIR / "users_features_v2.csv"
    if verify and sha256_of(path) != config.DATA_HASHES["features_v2"]:
        raise ValueError("features_v2 hash mismatch")
    df = pd.read_csv(path)
    df = df.drop(columns=["num_sessions"])  # all-zero in this export
    for col in ("min_to_first_level_start", "min_to_first_post_score",
                "min_to_first_level_complete"):
        df[col] = df[col].fillna(config.NEVER_SENTINEL)
    return df


def load_v2(name: str, verify: bool = True) -> pd.DataFrame:
    base = load_dataset(name, verify=verify)
    feats = _features_v2(verify=verify)
    df = base.merge(feats, on="user_pseudo_id", how="left", validate="one_to_one")
    if df[config.V2_RAW_NUMERICAL].isna().any().any():
        raise ValueError(f"unmatched rows after v2 join for split {name!r}")

    # Derived ratios (guard divide-by-zero with +1 in the denominator).
    df["completion_rate"] = df["cnt_level_complete_quickplay"] / (df["cnt_level_start_quickplay"] + 1)
    df["reset_rate"] = df["cnt_level_reset_quickplay"] / (df["cnt_level_start_quickplay"] + 1)
    df["early_fraction_1h"] = df["cnt_events_first_1h"] / (df["cnt_events_total"] + 1)
    df["events_per_type"] = df["cnt_events_total"] / (df["num_distinct_event_types"] + 1)
    return df
