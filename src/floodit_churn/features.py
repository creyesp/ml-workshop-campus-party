"""Preprocesamiento de features (sklearn ColumnTransformer).

Migrado y limpiado de ``notebooks/src/transformer.py``:
- corregidos typos (``categorical``, docstrings),
- eliminado el ``print`` de debug del ``fit``,
- ``MostCommonCategories.transform`` ya no muta el array de entrada in-place,
- API de sklearn actualizada (``OneHotEncoder`` sin argumentos deprecados).

El preprocessor por defecto reproduce el usado por el modelo XGBoost ganador
(notebook ``3.3_modeling_xgb``): imputación + StandardScaler para numéricas,
imputación + OneHotEncoder para categóricas.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS


def build_preprocessor(
    numeric: bool = True,
    categorical: bool = True,
    most_common: float | None = None,
) -> ColumnTransformer:
    """Construye el ColumnTransformer de preprocesamiento.

    Args:
        numeric: incluir las columnas numéricas (imputación + escalado).
        categorical: incluir las columnas categóricas (imputación + one-hot).
        most_common: si se indica (0-1), reduce la cardinalidad de las
            categóricas agrupando las categorías poco frecuentes en "other".
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_steps: list = [("imputer", SimpleImputer(strategy="most_frequent"))]
    if most_common:
        categorical_steps.append(("most_common", MostCommonCategories(thr=most_common)))
    categorical_steps.append(
        ("onehot", OneHotEncoder(categories="auto", handle_unknown="ignore"))
    )
    categorical_transformer = Pipeline(steps=categorical_steps)

    return ColumnTransformer(
        transformers=[
            (
                "numeric_features",
                numeric_transformer if numeric else "drop",
                NUMERICAL_COLUMNS,
            ),
            (
                "categorical_features",
                categorical_transformer if categorical else "drop",
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
    )


class MostCommonCategories(BaseEstimator, TransformerMixin):
    """Reduce la cardinalidad de features categóricas.

    Agrupa en la categoría ``"other"`` los valores cuya frecuencia acumulada
    supera el umbral ``thr``.
    """

    def __init__(self, thr: float = 0.8):
        self.thr = thr
        self.common_categories_: dict[int, np.ndarray] = {}

    def fit(self, X, y=None):
        for col in range(X.shape[1]):
            unique_elements, counts_elements = np.unique(X[:, col], return_counts=True)
            selector = np.cumsum(counts_elements) / counts_elements.sum() < self.thr
            self.common_categories_[col] = unique_elements[selector]
        return self

    def transform(self, X, y=None):
        # No mutar la entrada: trabajar sobre una copia.
        X = np.array(X, dtype=object, copy=True)
        for col in range(X.shape[1]):
            mask = np.isin(X[:, col], self.common_categories_[col], invert=True)
            X[mask, col] = "other"
        return X
