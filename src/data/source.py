import pandas as pd

from src.config.settings import (
    ALL_FEATURE_COLUMNS,
    DATA_DIR,
    LABEL_COLUMN,
    TEST_FILE,
    TRAIN_FILE,
)


def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / TRAIN_FILE)
    return df


def load_train_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_DIR / TRAIN_FILE)
    x = df[ALL_FEATURE_COLUMNS].copy()
    y = df[LABEL_COLUMN].copy()
    return x, y


def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_DIR / TEST_FILE)
    x = df[ALL_FEATURE_COLUMNS].copy()
    y = df[LABEL_COLUMN].copy()
    return x, y


def load_inference_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = [c for c in ALL_FEATURE_COLUMNS if c in df.columns]
    return df[cols].copy()
