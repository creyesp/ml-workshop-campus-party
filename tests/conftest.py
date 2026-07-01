"""Fixtures compartidos: un dataset sintético mínimo con el esquema esperado."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from floodit_churn.config import (
    CATEGORICAL_COLUMNS,
    IGNORE_COLUMNS,
    LABEL_COLUMN,
    NUMERICAL_COLUMNS,
)


def _make_df(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, object] = {}
    # Columnas ignoradas / id
    data["user_first_engagement"] = ["2021-01-01"] * n
    data["user_pseudo_id"] = [f"user_{i:04d}" for i in range(n)]
    data["is_enable"] = [1] * n
    data["bounced"] = [0] * n
    # Categóricas
    data["country_name"] = rng.choice(["Uruguay", "Chile", "India"], n)
    data["device_os"] = rng.choice(["ANDROID", "IOS"], n)
    data["device_lang"] = rng.choice(["es-uy", "en-us"], n)
    # Numéricas
    for col in NUMERICAL_COLUMNS:
        data[col] = rng.integers(0, 20, n)
    # Label correlacionado con engagement para que el modelo aprenda algo
    proba = 1 / (1 + np.exp(-(5 - data["cnt_user_engagement"])))
    data[LABEL_COLUMN] = (rng.random(n) < proba).astype(int)
    df = pd.DataFrame(data)
    # Garantiza ambas clases presentes
    df.loc[0, LABEL_COLUMN] = 0
    df.loc[1, LABEL_COLUMN] = 1
    return df


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return _make_df()


@pytest.fixture
def sample_csv(tmp_path, sample_df) -> str:
    path = tmp_path / "sample.csv"
    sample_df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def expected_columns() -> list[str]:
    return IGNORE_COLUMNS + CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS + [LABEL_COLUMN]
