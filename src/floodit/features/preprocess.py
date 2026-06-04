"""Pure preprocessing factory.

Ported and cleaned from ``notebooks/src/transformer.py`` (frozen legacy). A
``ColumnTransformer`` that, at prediction time (spec §3), uses only the
``cnt_*`` counts and the three categorical attributes; identity/timing columns
in ``IGNORE_COLUMNS`` are dropped so they can never become features.

Numeric: impute missing with 0 (a missing count means the event never fired),
then standardize. Categorical: impute most-frequent, then one-hot with
``handle_unknown="ignore"`` so unseen categories at serving time are tolerated.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from floodit.config import CATEGORICAL_COLUMNS, IGNORE_COLUMNS, NUMERICAL_COLUMNS


def make_preprocessor(numerical_columns, categorical_columns) -> ColumnTransformer:
    """Preprocessor over an explicit column set (used to compare feature sets).

    Same transforms as ``build_preprocessor`` but the numeric/categorical column
    lists are passed in, and any other column is dropped via remainder.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # Dense output so tree models that reject sparse (HistGB) work too.
            ("onehot", OneHotEncoder(categories="auto", handle_unknown="ignore",
                                     sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric_features", numeric_transformer, list(numerical_columns)),
            ("categorical_features", categorical_transformer, list(categorical_columns)),
        ],
        remainder="drop",
    )


def build_preprocessor(numeric: bool = True, categorical: bool = True) -> ColumnTransformer:
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
    return ColumnTransformer(
        transformers=[
            ("numeric_features", numeric_transformer if numeric else "drop", NUMERICAL_COLUMNS),
            ("categorical_features", categorical_transformer if categorical else "drop", CATEGORICAL_COLUMNS),
            ("ignore_features", "drop", IGNORE_COLUMNS),
        ],
    )
