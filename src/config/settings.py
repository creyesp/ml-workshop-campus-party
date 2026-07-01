from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TRAIN_FILE = "users_train.csv"
TEST_FILE = "users_test.csv"

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
LABEL_COLUMN = "churned"

ALL_FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS
ALL_COLUMNS = ALL_FEATURE_COLUMNS + [LABEL_COLUMN]

RANDOM_STATE = 42
TEST_SIZE = 0.2

RUN_ID_FORMAT = "%Y%m%d%H%M"

DEFAULT_TRAIN_PARAMS = {
    "LogisticRegression": {
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs",
    },
    "RandomForestClassifier": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_leaf": 10,
    },
    "GradientBoostingClassifier": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
    },
    "XGBClassifier": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "eval_metric": "logloss",
        "use_label_encoder": False,
    },
}

MODEL_REGISTRY = {
    "LogisticRegression": "sklearn.linear_model.LogisticRegression",
    "RandomForestClassifier": "sklearn.ensemble.RandomForestClassifier",
    "GradientBoostingClassifier": "sklearn.ensemble.GradientBoostingClassifier",
    "XGBClassifier": "xgboost.XGBClassifier",
}

HPO_DEFAULT_N_TRIALS = 20
HPO_DEFAULT_METRIC = "roc_auc"
HPO_CV_FOLDS = 3

HPO_SEARCH_SPACE = {
    "LogisticRegression": {
        "C": {"low": 0.01, "high": 10.0, "type": "float", "log": True},
        "solver": {"choices": ["lbfgs", "liblinear"], "type": "categorical"},
    },
    "RandomForestClassifier": {
        "n_estimators": {"low": 50, "high": 300, "type": "int"},
        "max_depth": {"low": 3, "high": 20, "type": "int"},
        "min_samples_leaf": {"low": 1, "high": 20, "type": "int"},
    },
    "GradientBoostingClassifier": {
        "n_estimators": {"low": 50, "high": 300, "type": "int"},
        "learning_rate": {"low": 0.01, "high": 0.3, "type": "float", "log": True},
        "max_depth": {"low": 2, "high": 6, "type": "int"},
    },
    "XGBClassifier": {
        "n_estimators": {"low": 50, "high": 300, "type": "int"},
        "learning_rate": {"low": 0.01, "high": 0.3, "type": "float", "log": True},
        "max_depth": {"low": 2, "high": 6, "type": "int"},
        "subsample": {"low": 0.6, "high": 1.0, "type": "float"},
    },
}
