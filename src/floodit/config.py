"""Central configuration: columns, seed, capped parallelism, paths.

Column lists ported verbatim from ``notebooks/src/config.py`` (frozen legacy).
"""
import os
from pathlib import Path

# --- Reproducibility -------------------------------------------------------
RANDOM_SEED = 42

# --- Parallelism -----------------------------------------------------------
# Never -1. Cap parallelism so other processes on the machine keep CPU.
# Override with FLOODIT_N_JOBS, but the value is clamped to [1, 4].
N_JOBS = max(1, min(4, int(os.environ.get("FLOODIT_N_JOBS", "2"))))

# --- Columns ---------------------------------------------------------------
CATEGORICAL_COLUMNS = [
    "country_name",
    "device_os",
    "device_lang",
]
NUMERICAL_COLUMNS = [
    "cnt_user_engagement",
    "cnt_level_start_quickplay",
    "cnt_level_end_quickplay",
    "cnt_level_complete_quickplay",
    "cnt_level_reset_quickplay",
    "cnt_post_score",
    "cnt_spend_virtual_currency",
    "cnt_ad_reward",
    "cnt_challenge_a_friend",
    "cnt_completed_5_levels",
    "cnt_use_extra_steps",
]
# Identity / timing / upstream-exclusion columns that must never be features.
IGNORE_COLUMNS = [
    "user_first_engagement",
    "user_pseudo_id",
    "is_enable",
    "bounced",
]
LABEL_COLUMN = "churned"
ALL_COLUMNS = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS + [LABEL_COLUMN]

# --- Paths -----------------------------------------------------------------
# config.py is src/floodit/config.py -> repo root is parents[2].
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
MLRUNS_DIR = REPO_ROOT / "mlruns"

# Dataset sha256 digests, pinned via experiments/00_pin_data.py (Task 2).
DATA_HASHES: dict[str, str] = {
    "train": "1dea2802d7db3fafa823e9a904ed1b569635d7911807039b7382245d7008ce14",
    "test": "57a5adf560c50432caa13fb1a5a9fce913603e4fde77c2cb2ed889282b9723bb",
    "raw": "300b88bb463166e5d4818a0b3284b0344b8ef181d0149d2f1b5b6f3105190cf0",
    # Engineered features v2 (data/queries/flood_it_features_v2.sql), keyed by user.
    "features_v2": "d07811a83dd04960808159052fc3f2f13f83cdbd3712389ddf533f36d6275aa2",
    # Session-level features v3 (data/queries/flood_it_features_v3_sessions.sql).
    "features_v3": "0969ed265b5d903b04782cd28adf9101d5c14f8bf6af5645810c161bbfdcd2df",
    # Navigation-sequence features v4 (data/queries/flood_it_features_v4_navigation.sql).
    "features_v4": "d652d10455fc2024cb7cd726d805cff2c7b6f90c790ef81a992d3c5a58126f6e",
    # Markov / sequence-order features v5 (data/queries/flood_it_features_v5_markov.sql).
    "features_v5": "07fb27e1b2b79732aadff815e59b91a4763a9a922e14614e3f9829a3a33f4c90",
}

# --- Feature set v2 --------------------------------------------------------
# New BigQuery-derived features (num_sessions dropped: all-zero in this export).
# min_to_first_* nulls ("never did it in window") are imputed to NEVER_SENTINEL.
NEVER_SENTINEL = 1441  # minutes: one past the 24h observation window
V2_RAW_NUMERICAL = [
    "cnt_events_total",
    "cnt_events_first_1h",
    "cnt_events_first_6h",
    "num_distinct_event_types",
    "min_to_first_level_start",
    "min_to_first_post_score",
    "min_to_first_level_complete",
]
# Ratio/derived features computed in load_v2 from existing + v2 counts.
V2_DERIVED_NUMERICAL = [
    "completion_rate",
    "reset_rate",
    "early_fraction_1h",
    "events_per_type",
]
V2_NUMERICAL_COLUMNS = NUMERICAL_COLUMNS + V2_RAW_NUMERICAL + V2_DERIVED_NUMERICAL

# --- Feature set v3: session-level (time-gap sessionization) ---------------
# Derived in BigQuery (no ga_session_id in this export). min_to_second_session
# null = never returned in window -> NEVER_SENTINEL; mean_intra_gap null = single
# event -> 0.
V3_RAW_SESSION = [
    "sess_num_sessions",
    "sess_events_per_session",
    "sess_total_engagement_sec",
    "sess_engagement_first_1h_sec",
    "sess_mean_engagement_msec",
    "sess_num_distinct_screens",
    "sess_mean_intra_gap_s",
    "sess_min_to_second_session",
]
V3_DERIVED_SESSION = [
    "sess_returned",            # 1 if >=2 sessions in window
    "sess_engagement_early_frac",  # first-1h engagement / total engagement
]
V3_NUMERICAL_COLUMNS = V2_NUMERICAL_COLUMNS + V3_RAW_SESSION + V3_DERIVED_SESSION

# --- Feature set v4: navigation sequences (logical screen categories) -------
# Screen classes mapped to platform-agnostic categories in BigQuery. Users with
# no screen views in window (30 of train) get all-zero nav features.
V4_RAW_NAV = [
    "nav_screen_views",
    "nav_distinct_categories",
    "nav_transitions",
    "nav_distinct_transitions",
    "nav_self_loops",
    "nav_cnt_game_over",
    "nav_cnt_shop",
    "nav_cnt_steps",
    "nav_cnt_ad",
    "nav_cnt_level_select",
]
V4_DERIVED_NAV = [
    "nav_transition_rate",   # transitions / screen views
    "nav_revisit_rate",      # self-loops / transitions
    "nav_game_over_rate",    # game_over views / screen views
    "nav_reached_shop",      # 1 if visited shop
    "nav_reached_steps",     # 1 if hit out-of-steps/extra-steps
]
V4_NUMERICAL_COLUMNS = V3_NUMERICAL_COLUMNS + V4_RAW_NAV + V4_DERIVED_NAV

# --- Feature set v5: Markov / sequence-ORDER (first-order transition probs) --
# v4 (navigation COUNTS) added nothing. v5 tests whether transition ORDER carries
# orthogonal signal: per-source-category transition PROBABILITIES P(cur|prev) =
# count(prev>cur) / count(prev>*), built in load_v5 from raw edge counts. These
# are scale-free (probabilities), unlike the v4 counts. A few 2-gram presence
# flags are added. Users with no transitions in window -> all-zero (no order).
# Raw edge-count columns from the query (denominators + numerators).
V5_RAW_MARKOV = [
    "mk_total_transitions",
    "mk_from_game", "mk_from_game_over", "mk_from_menu",
    "mk_from_level_select", "mk_from_steps",
    "mk_game__game", "mk_game__game_over", "mk_game__menu",
    "mk_game__steps", "mk_game__shop", "mk_game__level_select",
    "mk_game_over__game", "mk_game_over__menu", "mk_game_over__level_select",
    "mk_game_over__shop",
    "mk_menu__game", "mk_menu__level_select",
    "mk_level_select__game",
    "mk_steps__shop", "mk_steps__game",
]
# First-order Markov transition PROBABILITIES P(cur|prev) (the ORDER signal).
V5_PROB_MARKOV = [
    "p_game__game", "p_game__game_over", "p_game__menu",
    "p_game__steps", "p_game__shop", "p_game__level_select",
    "p_game_over__game", "p_game_over__menu", "p_game_over__level_select",
    "p_game_over__shop",
    "p_menu__game", "p_menu__level_select",
    "p_level_select__game",
    "p_steps__shop", "p_steps__game",
]
# 2-gram presence flags (did this directed transition occur at all in window).
V5_BIGRAM_FLAGS = [
    "ng_game_to_game_over",   # bounced from play to game-over
    "ng_game_over_to_game",   # retried after game-over (re-engagement)
    "ng_steps_to_shop",       # out-of-steps -> shop (monetization funnel)
    "ng_game_to_shop",        # play -> shop
]
V5_MARKOV_COLUMNS = V5_PROB_MARKOV + V5_BIGRAM_FLAGS
V5_NUMERICAL_COLUMNS = V3_NUMERICAL_COLUMNS + V5_MARKOV_COLUMNS
