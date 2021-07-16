from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .config import NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS, IGNORE_COLUMNS


def preprocessor(numeric=True, categical=True):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(categories="auto", handle_unknown="ignore"),
            ),
        ]
    )

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
            ("ignore_features", "drop", IGNORE_COLUMNS),
        ],
        n_jobs=-1,
    )
    return preprocessor
