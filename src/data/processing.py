"""
Saneamiento y split de datos.

Contiene la limpieza mínima previa al feature engineering (tipado de columnas,
descarte de identificadores y columnas con fuga, deduplicación) y la partición
estratificada train/test. El feature engineering (imputación, escalado, one-hot)
NO vive aquí: se resuelve en ``src.features.build`` como pipeline de sklearn.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.settings import get_settings


def sanitize_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza mínima y determinista del dataset crudo.

    - Deduplica por identificador de usuario si está presente.
    - Fuerza a numérico las columnas de conteo de eventos.
    - Castea la etiqueta a entero {0, 1}.
    - Descarta filas sin etiqueta.

    No imputa ni escala: eso es responsabilidad del pipeline de features, que se
    ajusta solo con train para evitar fuga de información.
    """
    config = get_settings().data
    clean = dataframe.copy()

    if "user_pseudo_id" in clean.columns:
        clean = clean.drop_duplicates(subset="user_pseudo_id")

    for column in config.numeric_features:
        if column in clean.columns:
            clean[column] = pd.to_numeric(clean[column], errors="coerce")

    label = config.label_column
    if label not in clean.columns:
        raise KeyError(f"La columna etiqueta '{label}' no está en el dataset.")
    clean = clean.dropna(subset=[label])
    clean[label] = clean[label].astype(int)

    if get_settings().verbose:
        print(f"[data] Saneamiento: {dataframe.shape} -> {clean.shape}")
    return clean.reset_index(drop=True)


def split_data(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide el dataset en train y test de forma estratificada por la etiqueta.

    El tamaño de test y la semilla provienen de la configuración.
    """
    config = get_settings().data
    label = config.label_column
    train_df, test_df = train_test_split(
        dataframe,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=dataframe[label],
    )
    if get_settings().verbose:
        print(f"[data] Split -> train {train_df.shape}, test {test_df.shape}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def split_features_target(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa las columnas de features y la etiqueta objetivo."""
    config = get_settings().data
    features = dataframe[config.feature_columns]
    target = dataframe[config.label_column]
    return features, target
