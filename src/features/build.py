import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.settings import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS


class MostCommonCategories(BaseEstimator, TransformerMixin):
    """Reduce cardinalidad de categorías agrupando las menos frecuentes como 'other'."""

    def __init__(self, thr: float = 0.8):
        self.thr = thr
        self.common_categories_: dict[int, np.ndarray] = {}

    def fit(self, x, y=None):
        for col in range(x.shape[1]):
            unique_elements, counts_elements = np.unique(x[:, col], return_counts=True)
            order = np.argsort(-counts_elements)
            unique_elements = unique_elements[order]
            counts_elements = counts_elements[order]
            cumulative = np.cumsum(counts_elements) / counts_elements.sum()
            selector = cumulative < self.thr
            self.common_categories_[col] = unique_elements[selector]
        return self

    def transform(self, x, y=None):
        x = x.copy()
        for col in range(x.shape[1]):
            mask = ~np.isin(x[:, col], self.common_categories_[col])
            x[mask, col] = "other"
        return x


def build_preprocessor(
    most_common_thr: float | None = None,
) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ]
    if most_common_thr is not None:
        categorical_steps.append(
            ("most_common", MostCommonCategories(thr=most_common_thr))
        )
    categorical_steps.append(
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    )
    categorical_transformer = Pipeline(steps=categorical_steps)
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERICAL_COLUMNS),
            ("categorical", categorical_transformer, CATEGORICAL_COLUMNS),
        ],
    )
    return preprocessor
