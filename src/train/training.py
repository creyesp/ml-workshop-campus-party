import importlib

import pandas as pd
from sklearn.pipeline import Pipeline

from src.config.settings import (
    DEFAULT_TRAIN_PARAMS,
    MODEL_REGISTRY,
    RANDOM_STATE,
)
from src.features.build import build_preprocessor


def _resolve_model_class(model_name: str):
    module_path, class_name = MODEL_REGISTRY[model_name].rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def build_pipeline(
    model_name: str,
    model_params: dict | None = None,
    most_common_thr: float | None = None,
) -> Pipeline:
    if model_params is None:
        model_params = DEFAULT_TRAIN_PARAMS.get(model_name, {}).copy()
    params = {**model_params}
    if "random_state" not in str(params.get("random_state", type(None))).lower():
        params["random_state"] = RANDOM_STATE
    model_class = _resolve_model_class(model_name)
    model = model_class(**params)
    preprocessor = build_preprocessor(most_common_thr=most_common_thr)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )
    return pipeline


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    model_params: dict | None = None,
    most_common_thr: float | None = None,
) -> Pipeline:
    pipeline = build_pipeline(
        model_name=model_name,
        model_params=model_params,
        most_common_thr=most_common_thr,
    )
    pipeline.fit(x_train, y_train)
    return pipeline
