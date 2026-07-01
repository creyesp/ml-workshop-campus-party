"""
Construcción y entrenamiento de modelos de churn.

Todos los modelos son clasificadores. El artefacto entrenado es un único
``Pipeline`` de sklearn que encadena el feature engineering y el clasificador, de
modo que features + modelo viajan juntos y se ajustan/aplican de forma consistente.

Familias de salida soportadas:
- ``point``: predicción de clase (churn sí/no).
- ``probabilistic``: probabilidad de churn (``predict_proba``).

Ambas familias comparten el mismo estimador entrenado; se diferencian en cómo se
consume la salida (ver ``src.train.inference``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config.settings import (
    DEFAULT_GRADIENT_BOOSTING_PARAMS,
    DEFAULT_LOGISTIC_PARAMS,
    DEFAULT_RANDOM_FOREST_PARAMS,
    DEFAULT_XGBOOST_PARAMS,
    get_settings,
)
from src.features.build import build_feature_pipeline

# Parámetros por defecto por algoritmo (definidos en settings, no hardcodeados aquí).
_DEFAULT_PARAMS: dict[str, dict] = {
    "logistic": DEFAULT_LOGISTIC_PARAMS,
    "random_forest": DEFAULT_RANDOM_FOREST_PARAMS,
    "gradient_boosting": DEFAULT_GRADIENT_BOOSTING_PARAMS,
    "xgboost": DEFAULT_XGBOOST_PARAMS,
}


def _scale_pos_weight(y_train: pd.Series) -> float:
    """Ratio negativos/positivos para compensar el desbalance en XGBoost."""
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    return negatives / positives if positives else 1.0


def build_classifier(
    model_name: str,
    y_train: pd.Series | None = None,
    **overrides,
) -> ClassifierMixin:
    """
    Construye un clasificador sin entrenar.

    Args:
        model_name: uno de ``settings.training.available_models``.
        y_train: etiqueta de entrenamiento; se usa para calcular el balanceo de
            clases en XGBoost (``scale_pos_weight``).
        **overrides: hiperparámetros que sobrescriben los defaults.
    """
    if model_name not in _DEFAULT_PARAMS:
        raise ValueError(
            f"Modelo desconocido: {model_name}. Disponibles: {list(_DEFAULT_PARAMS)}"
        )

    random_state = get_settings().random_seed
    params = {**_DEFAULT_PARAMS[model_name], **overrides}

    if model_name == "logistic":
        return LogisticRegression(random_state=random_state, **params)
    if model_name == "random_forest":
        return RandomForestClassifier(random_state=random_state, **params)
    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state, **params)
    if model_name == "xgboost":
        if "scale_pos_weight" not in params and y_train is not None:
            params["scale_pos_weight"] = _scale_pos_weight(y_train)
        return XGBClassifier(random_state=random_state, **params)

    raise ValueError(f"Modelo no implementado: {model_name}")  # pragma: no cover


def build_model(
    model_name: str,
    y_train: pd.Series | None = None,
    **overrides,
) -> Pipeline:
    """
    Construye el pipeline completo (features + clasificador) sin entrenar.

    Devuelve un único artefacto de inferencia: al hacer ``fit`` sobre el DataFrame
    crudo, ajusta el feature engineering y el modelo en un solo paso.
    """
    classifier = build_classifier(model_name, y_train=y_train, **overrides)
    return Pipeline(
        steps=[
            ("features", build_feature_pipeline()),
            ("classifier", classifier),
        ]
    )


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    model_name: str,
    model_params: dict | None = None,
) -> Pipeline:
    """
    Entrena el pipeline completo sobre los datos de entrenamiento.

    Args:
        features: DataFrame con las columnas de features crudas.
        target: etiqueta de entrenamiento.
        model_name: algoritmo a usar.
        model_params: hiperparámetros que sobrescriben los defaults.

    Returns:
        Pipeline entrenado (features + clasificador).
    """
    model = build_model(model_name, y_train=target, **(model_params or {}))
    model.fit(features, target)
    if get_settings().verbose:
        print(f"[train] Modelo '{model_name}' entrenado sobre {len(features)} muestras")
    return model


def predict_proba(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Probabilidad de churn (clase positiva)."""
    return model.predict_proba(features)[:, 1]


def predict_label(
    model: Pipeline,
    features: pd.DataFrame,
    threshold: float | None = None,
) -> np.ndarray:
    """Predicción de clase aplicando el umbral de decisión configurado."""
    if threshold is None:
        threshold = get_settings().training.decision_threshold
    return (predict_proba(model, features) >= threshold).astype(int)
