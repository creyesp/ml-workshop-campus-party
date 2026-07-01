"""
Construcción del pipeline de features.

El feature engineering se expresa como un ``ColumnTransformer`` explícito, no como
funciones sueltas de pandas. Esto garantiza que la transformación se ajuste solo con
train (``fit``) y se reutilice idéntica en test e inferencia (``transform``), evitando
fuga de información.

- Numéricas: imputación por constante + estandarización.
- Categóricas: imputación por más frecuente + one-hot (ignora categorías nuevas).
- Cualquier otra columna (ids, etiqueta) se descarta vía ``remainder="drop"``.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config.settings import get_settings


def _as_float(array: np.ndarray) -> np.ndarray:
    """Castea las columnas numéricas a float.

    Definida a nivel de módulo (no lambda) para que el artefacto sea serializable
    con joblib. Además garantiza que el imputador reciba siempre dtype float,
    independientemente de si la entrada llega como int (evita el chequeo estricto
    de casting de sklearn).
    """
    return np.asarray(array, dtype=float)


def _numeric_pipeline() -> Pipeline:
    config = get_settings().features
    steps = [
        ("cast", FunctionTransformer(_as_float, feature_names_out="one-to-one")),
        (
            "imputer",
            SimpleImputer(
                strategy="constant", fill_value=config.numeric_imputer_fill_value
            ),
        ),
    ]
    if config.scale_numeric:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps=steps)


def _categorical_pipeline() -> Pipeline:
    config = get_settings().features
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=config.categorical_imputer_strategy)),
            (
                "onehot",
                OneHotEncoder(handle_unknown=config.onehot_handle_unknown),
            ),
        ]
    )


def build_feature_pipeline() -> ColumnTransformer:
    """
    Devuelve el ``ColumnTransformer`` de features SIN ajustar.

    El ajuste ocurre cuando el pipeline completo (features + clasificador) hace
    ``fit`` sobre los datos de entrenamiento.
    """
    data_config = get_settings().data
    return ColumnTransformer(
        transformers=[
            ("numeric", _numeric_pipeline(), data_config.numeric_features),
            ("categorical", _categorical_pipeline(), data_config.categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
