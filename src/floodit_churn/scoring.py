"""Batch scoring: puntúa un CSV de usuarios y escribe predicciones a un CSV local."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DEFAULT_THRESHOLD, ID_COLUMN
from .data import load_dataset
from .model import load_metadata, load_model


def score_dataframe(
    df: pd.DataFrame,
    model,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Puntúa un DataFrame y devuelve id + probabilidad + predicción binaria.

    Args:
        df: usuarios a puntuar (con las columnas de features).
        model: pipeline entrenado (preprocessor + clasificador).
        threshold: umbral de decisión (default 0.5, configurable).
    """
    proba = model.predict_proba(df)[:, 1]
    result = pd.DataFrame()
    if ID_COLUMN in df.columns:
        result[ID_COLUMN] = df[ID_COLUMN].values
    result["churn_proba"] = proba
    result["churn_pred"] = (proba >= threshold).astype(int)
    return result


def score_csv(
    input_csv: str | Path,
    model_dir: str | Path,
    output_csv: str | Path,
    threshold: float | None = None,
) -> Path:
    """Puntúa ``input_csv`` con el modelo en ``model_dir`` y escribe ``output_csv``.

    Si ``threshold`` es None, usa el ``default_threshold`` de la metadata del
    artefacto y, en su defecto, el default global (0.5).
    """
    model = load_model(model_dir)
    if threshold is None:
        threshold = load_metadata(model_dir).get("default_threshold", DEFAULT_THRESHOLD)

    df = load_dataset(input_csv, require_label=False)
    result = score_dataframe(df, model, threshold=threshold)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return output_csv
