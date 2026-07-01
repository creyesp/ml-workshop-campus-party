"""Fixtures compartidas para los tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.config.settings import get_settings


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Dataset sintético pequeño con el esquema esperado por el pipeline."""
    rng = np.random.default_rng(0)
    n = 200
    data = get_settings().data

    frame = {"user_pseudo_id": [f"u{i}" for i in range(n)]}
    for column in data.numeric_features:
        frame[column] = rng.integers(0, 50, size=n)
    frame["country_name"] = rng.choice(["United States", "Brazil", "India"], size=n)
    frame["device_os"] = rng.choice(["ANDROID", "IOS"], size=n)
    frame["device_lang"] = rng.choice(["en-us", "pt-br", "es-es"], size=n)

    # Señal débil para que los modelos aprendan algo por encima del azar.
    logit = 0.05 * frame["cnt_user_engagement"] - 1.0
    prob = 1 / (1 + np.exp(-logit))
    frame[data.label_column] = (rng.random(n) < prob).astype(int)
    return pd.DataFrame(frame)
