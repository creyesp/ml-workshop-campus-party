"""Load the v3 dataset joined with Markov sequence-ORDER v5 features.

Same train/test rows as ``load_v3``. v4 (navigation COUNTS) added nothing; v5
tests whether the transition ORDER carries orthogonal signal. Raw directed-edge
counts come from ``data/queries/flood_it_features_v5_markov.sql`` (same logical
screen-category mapping and 24h window as v4). This loader converts them into
first-order Markov transition PROBABILITIES P(cur|prev) — normalized within each
SOURCE category — plus a few 2-gram presence flags.

P(cur|prev) = count(prev>cur) / count(prev>*). Scale-free, unlike v4 counts. A
user with no transitions out of a given source category gets 0 for every edge
from it (no order information), so all probability features impute to 0.

LEAKAGE GUARD: window-bounded edge counts; no last-event time is encoded.
"""
import pandas as pd

from floodit import config
from floodit.data.load import sha256_of
from floodit.data.load_v3 import load_v3

# (probability column, numerator edge count, source-category denominator).
_PROB_SPEC = [
    ("p_game__game",              "mk_game__game",              "mk_from_game"),
    ("p_game__game_over",         "mk_game__game_over",         "mk_from_game"),
    ("p_game__menu",              "mk_game__menu",              "mk_from_game"),
    ("p_game__steps",             "mk_game__steps",             "mk_from_game"),
    ("p_game__shop",              "mk_game__shop",              "mk_from_game"),
    ("p_game__level_select",      "mk_game__level_select",      "mk_from_game"),
    ("p_game_over__game",         "mk_game_over__game",         "mk_from_game_over"),
    ("p_game_over__menu",         "mk_game_over__menu",         "mk_from_game_over"),
    ("p_game_over__level_select", "mk_game_over__level_select", "mk_from_game_over"),
    ("p_game_over__shop",         "mk_game_over__shop",         "mk_from_game_over"),
    ("p_menu__game",              "mk_menu__game",              "mk_from_menu"),
    ("p_menu__level_select",      "mk_menu__level_select",      "mk_from_menu"),
    ("p_level_select__game",      "mk_level_select__game",      "mk_from_level_select"),
    ("p_steps__shop",             "mk_steps__shop",             "mk_from_steps"),
    ("p_steps__game",             "mk_steps__game",             "mk_from_steps"),
]
# (flag column, raw edge-count column).
_FLAG_SPEC = [
    ("ng_game_to_game_over", "mk_game__game_over"),
    ("ng_game_over_to_game", "mk_game_over__game"),
    ("ng_steps_to_shop",     "mk_steps__shop"),
    ("ng_game_to_shop",      "mk_game__shop"),
]


def _features_v5(verify: bool = True) -> pd.DataFrame:
    path = config.DATA_DIR / "users_features_v5_markov.csv"
    if verify and sha256_of(path) != config.DATA_HASHES["features_v5"]:
        raise ValueError("features_v5 hash mismatch")
    return pd.read_csv(path)


def load_v5(name: str, verify: bool = True) -> pd.DataFrame:
    base = load_v3(name, verify=verify)
    feats = _features_v5(verify=verify)
    df = base.merge(feats, on="user_pseudo_id", how="left", validate="one_to_one")
    # Users with no transitions in window -> no order information at all.
    df[config.V5_RAW_MARKOV] = df[config.V5_RAW_MARKOV].fillna(0)

    # First-order Markov transition probabilities P(cur|prev), normalized within
    # each source category. Zero denominator (never visited that source) -> 0.
    for prob_col, num_col, den_col in _PROB_SPEC:
        den = df[den_col]
        df[prob_col] = (df[num_col] / den.where(den > 0, 1)).where(den > 0, 0.0)

    # 2-gram presence flags.
    for flag_col, edge_col in _FLAG_SPEC:
        df[flag_col] = (df[edge_col] > 0).astype(int)

    return df
