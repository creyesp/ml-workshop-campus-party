"""
Módulo de ingeniería de características.
Construye el pipeline de preprocesamiento utilizando transformadores de scikit-learn.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.settings import (
    CATEGORICAL_COLUMNS,
    DEFAULT_CARDINALITY_THRESHOLD,
    IGNORE_COLUMNS,
    NUMERICAL_COLUMNS,
)


class MostCommonCategories(BaseEstimator, TransformerMixin):
    """
    Transformador personalizado para reducir la cardinalidad de características categóricas,
    manteniendo solo las categorías más comunes que acumulan una cierta proporción de
    los datos y reemplazando las restantes por 'other'.
    """

    def __init__(self, thr: float = DEFAULT_CARDINALITY_THRESHOLD):
        """
        Inicializa el transformador.

        Args:
            thr (float): Umbral de proporción acumulada (0.0 a 1.0) para conservar categorías.
        """
        self.thr = thr
        self.common_categories_ = {}

    def fit(self, X, y=None):
        """
        Ajusta el transformador identificando las categorías más comunes en cada columna.
        """
        # Convertir a numpy array si es necesario
        X_arr = np.array(X)
        self.common_categories_ = {}

        for col in range(X_arr.shape[1]):
            # Obtener elementos únicos y sus conteos
            unique_elements, counts_elements = np.unique(
                X_arr[:, col].astype(str), return_counts=True
            )

            # Ordenar por frecuencia descendente (corrige el error en el notebook original)
            sort_idx = np.argsort(-counts_elements)
            unique_elements = unique_elements[sort_idx]
            counts_elements = counts_elements[sort_idx]

            # Calcular la proporción acumulada
            cumulative_proportions = np.cumsum(counts_elements) / counts_elements.sum()

            # Seleccionar las categorías que se quedan antes de superar el umbral
            # Siempre se mantiene al menos la categoría más común
            selector = cumulative_proportions <= self.thr
            if not np.any(selector):
                self.common_categories_[col] = unique_elements[:1]
            else:
                self.common_categories_[col] = unique_elements[selector]

        return self

    def transform(self, X, y=None):
        """
        Reemplaza las categorías poco comunes por 'other'.
        """
        X_arr = np.array(X, copy=True).astype(object)

        for col in range(X_arr.shape[1]):
            common = self.common_categories_.get(col, np.array([]))
            mask = np.isin(X_arr[:, col], common, invert=True)
            X_arr[mask, col] = "other"

        return X_arr


def get_preprocessor_pipeline(
    use_most_common: bool = False,
    cardinality_threshold: float = DEFAULT_CARDINALITY_THRESHOLD,
) -> ColumnTransformer:
    """
    Crea y retorna un pipeline de preprocesamiento estructurado con ColumnTransformer.

    Args:
        use_most_common (bool): Si es True, incluye MostCommonCategories en categóricas.
        cardinality_threshold (float): Umbral para MostCommonCategories.

    Returns:
        ColumnTransformer: Pipeline de preprocesamiento listo para fit/transform.
    """
    # Transformaciones para columnas numéricas
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )

    # Pasos para columnas categóricas
    categorical_steps = [("imputer", SimpleImputer(strategy="most_frequent"))]

    if use_most_common:
        categorical_steps.append(("most_common", MostCommonCategories(thr=cardinality_threshold)))

    # Configurar OneHotEncoder compatible con sklearn moderna
    categorical_steps.append(
        (
            "onehot",
            OneHotEncoder(categories="auto", handle_unknown="ignore", sparse_output=False),
        )
    )

    categorical_transformer = Pipeline(steps=categorical_steps)

    # ColumnTransformer para consolidar las características
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric_features",
                numeric_transformer,
                NUMERICAL_COLUMNS,
            ),
            (
                "categorical_features",
                categorical_transformer,
                CATEGORICAL_COLUMNS,
            ),
            (
                "ignore_features",
                "drop",
                IGNORE_COLUMNS,
            ),
        ],
        remainder="drop",
    )

    return preprocessor
