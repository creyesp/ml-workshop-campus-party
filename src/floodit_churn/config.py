"""Configuración central: columnas, rutas y constantes del proyecto.

Migrado de ``notebooks/src/config.py``. Es la única fuente de verdad para
nombres de columnas, rutas y semilla, tanto para entrenamiento como para scoring.
"""
from __future__ import annotations

from pathlib import Path

# --- Rutas (basadas en la raíz del repo, no en el cwd) ---
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

TRAIN_CSV = DATA_DIR / "users_train.csv"
TEST_CSV = DATA_DIR / "users_test.csv"

# --- Reproducibilidad ---
RANDOM_STATE = 42

# --- Esquema de features ---
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
# Columnas presentes en el CSV pero que no entran al modelo.
IGNORE_COLUMNS = [
    "user_first_engagement",
    "user_pseudo_id",
    "is_enable",
    "bounced",
]
# Identificador que se arrastra hasta el output del scoring.
ID_COLUMN = "user_pseudo_id"
LABEL_COLUMN = "churned"

# Columnas de features que consume el modelo (orden estable).
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS
# Todas las columnas esperadas en un CSV de entrenamiento (con label).
ALL_COLUMNS = IGNORE_COLUMNS + FEATURE_COLUMNS + [LABEL_COLUMN]

# Umbral de decisión por defecto (configurable en scoring).
DEFAULT_THRESHOLD = 0.5
