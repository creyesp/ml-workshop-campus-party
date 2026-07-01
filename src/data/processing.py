import pandas as pd

from src.config.settings import IGNORE_COLUMNS


def drop_ignore_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in IGNORE_COLUMNS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def ensure_no_leakage(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    if label_column in df.columns:
        return df.drop(columns=[label_column])
    return df
