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
}
