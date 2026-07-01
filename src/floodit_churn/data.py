"""Carga y validación de datasets."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .config import FEATURE_COLUMNS, LABEL_COLUMN


class SchemaError(ValueError):
    """Se levanta cuando un CSV no cumple el esquema esperado."""


def sha256(path: str | Path) -> str:
    """Hash SHA256 de un archivo (para pinnear datos en la metadata)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset(path: str | Path, require_label: bool = True) -> pd.DataFrame:
    """Carga un CSV y valida que contenga las columnas de features (y label).

    Args:
        path: ruta al CSV.
        require_label: si True, exige la columna ``churned`` (entrenamiento/eval).
            En scoring puede no estar presente.

    Raises:
        SchemaError: si faltan columnas de features o el label requerido.
    """
    df = pd.read_csv(path)

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Faltan columnas de features en {path}: {missing}")

    if require_label:
        if LABEL_COLUMN not in df.columns:
            raise SchemaError(f"Falta la columna label '{LABEL_COLUMN}' en {path}")
        if df[LABEL_COLUMN].isna().any():
            raise SchemaError(f"La columna label '{LABEL_COLUMN}' tiene valores nulos en {path}")

    return df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features (X) y label (y). Espera la columna label presente."""
    if LABEL_COLUMN not in df.columns:
        raise SchemaError(f"No se puede separar X/y: falta '{LABEL_COLUMN}'")
    x = df.drop(columns=[LABEL_COLUMN])
    y = df[LABEL_COLUMN]
    return x, y
