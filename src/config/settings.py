"""
Configuración centralizada para el pipeline de ML.
Contiene parámetros de datos, modelos, hiperparámetros por defecto,
espacios de búsqueda para HPO y rutas del proyecto.
"""

from pathlib import Path

# --- Rutas del Proyecto ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = BASE_DIR / "models"

# Crear directorios necesarios al importar el módulo
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Rutas de archivos de datos por defecto
DEFAULT_RAW_DATA_PATH = DATA_DIR / "users_raw.csv"
DEFAULT_TRAIN_DATA_PATH = DATA_DIR / "users_train.csv"
DEFAULT_TEST_DATA_PATH = DATA_DIR / "users_test.csv"
DEFAULT_NOT_YET_PATH = DATA_DIR / "users_not_yet.csv"

# --- Definición de Columnas ---
LABEL_COLUMN = "churned"

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

IGNORE_COLUMNS = [
    "user_first_engagement",
    "user_pseudo_id",
    "is_enable",
    "bounced",
]

# --- Configuración de Datos y Split ---
DEFAULT_TEST_SIZE = 0.1
DEFAULT_RANDOM_STATE = 42
DEFAULT_SHUFFLE = True

# Umbral de cardinalidad para el transformador MostCommonCategories
DEFAULT_CARDINALITY_THRESHOLD = 0.8

# --- Formato del Run ID para Registro de Artefactos ---
RUN_ID_FORMAT = "{timestamp}_{model_name}"

# --- Parámetros por Defecto para Modelos (Clasificación) ---
DEFAULT_LINEAR_PARAMS = {
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": DEFAULT_RANDOM_STATE,
}

DEFAULT_RF_PARAMS = {
    "class_weight": "balanced",
    "n_estimators": 100,
    "random_state": DEFAULT_RANDOM_STATE,
    "n_jobs": -1,
}

DEFAULT_GB_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "random_state": DEFAULT_RANDOM_STATE,
}

DEFAULT_XGB_PARAMS = {
    "eval_metric": "logloss",
    "random_state": DEFAULT_RANDOM_STATE,
    "n_jobs": -1,
}

# --- Parámetros por Defecto para Modelos (Regresión) ---
DEFAULT_REGRESSION_LINEAR_PARAMS = {
    "alpha": 1.0,
}

DEFAULT_REGRESSION_RF_PARAMS = {
    "n_estimators": 100,
    "random_state": DEFAULT_RANDOM_STATE,
    "n_jobs": -1,
}

DEFAULT_REGRESSION_GB_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "random_state": DEFAULT_RANDOM_STATE,
}

# --- Espacios de Búsqueda para HPO (Optuna) ---
HPO_NUM_TRIALS = 10
HPO_TIMEOUT_SECONDS = 600
HPO_N_SPLITS = 5

# Espacio de búsqueda para XGBoost Classifier (referencia documental)
XGB_HPO_SEARCH_SPACE = {
    "booster": ["gbtree", "gblinear", "dart"],
    "lambda": (1e-8, 1.0, "log"),
    "alpha": (1e-8, 1.0, "log"),
    "subsample": (0.2, 1.0, "uniform"),
    "colsample_bytree": (0.2, 1.0, "uniform"),
    "max_depth": (3, 7),
    "min_child_weight": (2, 10),
    "eta": (1e-8, 1.0, "log"),
    "gamma": (1e-8, 1.0, "log"),
    "grow_policy": ["depthwise", "lossguide"],
}
