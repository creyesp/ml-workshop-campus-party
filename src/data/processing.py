"""
Módulo de procesamiento y sanitización de datos.
Contiene la lógica para limpiar los datos crudos y dividirlos en conjuntos de
entrenamiento y validación/testeo de forma consistente.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.settings import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_SHUFFLE,
    DEFAULT_TEST_SIZE,
    LABEL_COLUMN,
)


def sanitize_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitiza y limpia el DataFrame crudo eliminando filas inválidas,
    valores nulos críticos en la variable objetivo y normalizando tipos de datos.

    Args:
        df (pd.DataFrame): DataFrame original.

    Returns:
        pd.DataFrame: DataFrame filtrado y saneado.
    """
    cleaned_df = df.copy()

    # Si la columna label tiene nulos, se descartan esas filas
    if LABEL_COLUMN in cleaned_df.columns:
        cleaned_df = cleaned_df.dropna(subset=[LABEL_COLUMN])
        cleaned_df[LABEL_COLUMN] = cleaned_df[LABEL_COLUMN].astype(int)

    return cleaned_df


def filter_by_business_rules(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica las reglas de negocio descritas en los notebooks.
    Separa el conjunto apto para entrenamiento/testeo (is_enable=1 y bounced=0)
    del conjunto de usuarios que aún no están listos (is_enable=0 y bounced=0).

    Args:
        df (pd.DataFrame): DataFrame completo.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - DataFrame apto para train/test.
            - DataFrame de usuarios 'not yet' (aún no listos).
    """
    # Evitar fallos si las columnas de reglas no existen
    if "is_enable" not in df.columns or "bounced" not in df.columns:
        return df.copy(), pd.DataFrame()

    selector_train_test = (df["is_enable"] == 1) & (df["bounced"] == 0)
    selector_not_yet = (df["is_enable"] == 0) & (df["bounced"] == 0)

    train_test_df = df.loc[selector_train_test].copy()
    not_yet_df = df.loc[selector_not_yet].copy()

    return train_test_df, not_yet_df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    shuffle: bool = DEFAULT_SHUFFLE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide un DataFrame en dos conjuntos de entrenamiento y prueba/validación.
    Realiza una división estratificada si la variable objetivo está presente.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        test_size (float): Proporción del conjunto de testeo.
        random_state (int): Semilla aleatoria.
        shuffle (bool): Mezclar antes de dividir.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames de train y test.
    """
    # Si la variable objetivo está presente, estratificar por ella
    stratify_col = None
    if LABEL_COLUMN in df.columns:
        # Verificar que haya al menos 2 clases y que la clase minoritaria tenga suficientes ejemplos
        class_counts = df[LABEL_COLUMN].value_counts()
        if len(class_counts) > 1 and class_counts.min() > 1:
            stratify_col = df[LABEL_COLUMN]

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify_col,
    )

    return train_df, test_df
