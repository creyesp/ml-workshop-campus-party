import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


from .config import NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS, IGNORE_COLUMNS


def preprocessor(
    numeric=True,
    categical=True,
    most_common=None,
):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )


    categorical_steps = [("imputer",SimpleImputer(strategy="most_frequent"))] 
    if most_common:
        categorical_steps += [("most_common", MostCommonCategories(thr=most_common))]
    categorical_steps += [("onehot", OneHotEncoder(categories="auto", handle_unknown="ignore"))]

    categorical_transformer = Pipeline(steps=categorical_steps)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric_features",
                numeric_transformer if numeric else "drop",
                NUMERICAL_COLUMNS,
            ),
            (
                "categorical_features",
                categorical_transformer if categical else "drop",
                CATEGORICAL_COLUMNS,
            ),
            (
                "ignore_features",
                "drop",
                IGNORE_COLUMNS,
            ),
        ],
        # n_jobs=-1,
    )
    return preprocessor


class MostCommonCategories(BaseEstimator, TransformerMixin):
    """Redece cadinality of categorical features"""

    def __init__(self, thr=0.8):
        self.thr = thr
        self.common_categories_ = {}

    def fit(self, X, y=None):
        print(
            type(X),
        )
        for col in range(X.shape[1]):
            unique_elements, counts_elements = np.unique(
                X[:, col], return_counts=True
            )
            selector = (
                np.cumsum(counts_elements) / counts_elements.sum() < self.thr
            )
            self.common_categories_[col] = unique_elements[selector]
        return self

    def transform(self, X, y=None):
        for col in range(X.shape[1]):
            mask = np.isin(X[:, col], self.common_categories_[col], invert=True)
            X[mask, col] = "other"

        return X
