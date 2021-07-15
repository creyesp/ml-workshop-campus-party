from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .config import NUMERICAL_COL, CATEGORICAL_COL, LABEL_COL


def preprocessor_numeric():
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor_num = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_COL),
            ("drop", "drop", CATEGORICAL_COL),
        ],
        n_jobs=-1,
    )
    return preprocessor_num


def preprocessor_full():
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(categories="auto", handle_unknown="ignore")),
        ]
    )

    preprocessor_full = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_COL),
            ("cat", categorical_transformer, CATEGORICAL_COL),
        ],
        n_jobs=-1,
    )
    return preprocessor_full