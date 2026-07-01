"""
Módulo de construcción y entrenamiento de modelos.
Contiene fábricas de modelos, estimadores personalizados para
cuantiles y wrappers de inferencia.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from src.config.settings import (
    DEFAULT_GB_PARAMS,
    DEFAULT_LINEAR_PARAMS,
    DEFAULT_REGRESSION_GB_PARAMS,
    DEFAULT_REGRESSION_LINEAR_PARAMS,
    DEFAULT_REGRESSION_RF_PARAMS,
    DEFAULT_RF_PARAMS,
    DEFAULT_XGB_PARAMS,
)
from src.features.build import get_preprocessor_pipeline


class QuantileRegressorWrapper(BaseEstimator, RegressorMixin):
    """
    Estimador personalizado que entrena múltiples modelos de regresión de gradiente
    para predecir un conjunto de cuantiles específicos y retornar intervalos.
    """

    def __init__(
        self,
        base_params: dict[str, Any],
        quantiles: list[float] = [0.1, 0.5, 0.9],
    ):
        """
        Args:
            base_params (Dict[str, Any]): Parámetros base para GradientBoostingRegressor.
            quantiles (List[float]): Cuantiles a estimar (ej. [0.1, 0.5, 0.9]).
        """
        self.base_params = base_params
        self.quantiles = quantiles
        self.models_ = {}

    def fit(self, X, y):
        """
        Entrena un modelo independiente para cada uno de los cuantiles definidos.
        """
        for q in self.quantiles:
            # Configurar el modelo para pérdida de cuantil específico
            params = self.base_params.copy()
            params["loss"] = "quantile"
            params["alpha"] = q
            model = GradientBoostingRegressor(**params)
            model.fit(X, y)
            self.models_[q] = model
        return self

    def predict(self, X) -> np.ndarray:
        """
        Genera predicciones para todos los cuantiles definidos.

        Returns:
            np.ndarray: Matriz de dimensiones (n_muestras, n_cuantiles)
                        con las predicciones para cada cuantil.
        """
        preds = []
        for q in self.quantiles:
            preds.append(self.models_[q].predict(X))
        # Formato: (n_samples, n_quantiles)
        return np.column_stack(preds)


class ProductionInferenceWrapper(BaseEstimator):
    """
    Wrapper final del modelo empaquetado para inferencia en producción.
    Encapsula el pipeline completo y garantiza que predict() devuelva
    la salida final de negocio formateada.

    Nota: los parámetros del __init__ deben coincidir exactamente con los
    atributos de instancia para que BaseEstimator.get_params() funcione
    correctamente con joblib y clonación sklearn.
    """

    def __init__(self, pipeline: Pipeline, task: str = "classification"):
        self.pipeline = pipeline
        self.task = task

    def fit(self, X, y):
        """Ajusta el pipeline interno completo."""
        self.pipeline.fit(X, y)
        return self

    def predict(self, X) -> pd.DataFrame:
        """
        Realiza la predicción y le da formato de salida de negocio.

        Args:
            X (pd.DataFrame): Datos crudos de entrada (sin preprocesar).

        Returns:
            pd.DataFrame: DataFrame formateado con los resultados del negocio.
        """
        if self.task == "classification":
            # Obtener predicciones puntuales y probabilidades
            preds = self.pipeline.predict(X)
            probs = self.pipeline.predict_proba(X)[:, 1]

            return pd.DataFrame(
                {
                    "prediction": preds,
                    "probability": probs,
                    "decision": np.where(preds == 1, "CHURN", "ACTIVE"),
                },
                index=X.index,
            )
        else:
            preds = self.pipeline.predict(X)

            # Regresión de cuantiles: columnas por cuantil
            if len(preds.shape) > 1 and preds.shape[1] > 1:
                return pd.DataFrame(
                    {
                        "quantile_10": preds[:, 0],
                        "quantile_50": preds[:, 1],
                        "quantile_90": preds[:, 2],
                        "interval_width": preds[:, 2] - preds[:, 0],
                    },
                    index=X.index,
                )
            else:
                return pd.DataFrame({"prediction": preds}, index=X.index)

    def predict_proba(self, X) -> np.ndarray:
        """Delega predict_proba al pipeline interno (solo clasificación)."""
        if self.task == "classification":
            return self.pipeline.predict_proba(X)
        raise AttributeError("Las tareas de regresión no admiten predict_proba.")


def build_estimator(
    model_name: str,
    task: str = "classification",
    hyperparameters: Optional[dict[str, Any]] = None,
) -> BaseEstimator:
    """
    Fábrica que construye el estimador (modelo) final con sus hiperparámetros.
    Implementa encapsulado de transformaciones para targets de regresión.

    Args:
        model_name (str): Nombre del modelo ('linear', 'rf', 'gb', 'xgb', 'quantile').
        task (str): Tarea ('classification' o 'regression').
        hyperparameters (Dict, opcional): Hiperparámetros que sobrescriben los defaults.

    Returns:
        BaseEstimator: Estimador de scikit-learn ajustado a la tarea.
    """
    params = hyperparameters if hyperparameters is not None else {}

    if task == "classification":
        if model_name == "linear":
            full_params = {**DEFAULT_LINEAR_PARAMS, **params}
            return LogisticRegression(**full_params)
        elif model_name == "rf":
            full_params = {**DEFAULT_RF_PARAMS, **params}
            return RandomForestClassifier(**full_params)
        elif model_name == "gb":
            full_params = {**DEFAULT_GB_PARAMS, **params}
            return GradientBoostingClassifier(**full_params)
        elif model_name == "xgb":
            full_params = {**DEFAULT_XGB_PARAMS, **params}
            return XGBClassifier(**full_params)
        else:
            raise ValueError(f"Modelo de clasificación no soportado: {model_name}")

    elif task == "regression":
        regressor = None
        if model_name == "linear":
            full_params = {**DEFAULT_REGRESSION_LINEAR_PARAMS, **params}
            regressor = Ridge(**full_params)
        elif model_name == "rf":
            full_params = {**DEFAULT_REGRESSION_RF_PARAMS, **params}
            regressor = RandomForestRegressor(**full_params)
        elif model_name == "gb":
            full_params = {**DEFAULT_REGRESSION_GB_PARAMS, **params}
            regressor = GradientBoostingRegressor(**full_params)
        elif model_name == "xgb":
            regressor = XGBRegressor(**params)
        elif model_name == "quantile":
            gb_params = {**DEFAULT_REGRESSION_GB_PARAMS, **params}
            return QuantileRegressorWrapper(base_params=gb_params)
        else:
            raise ValueError(f"Modelo de regresión no soportado: {model_name}")

        return TransformedTargetRegressor(
            regressor=regressor,
            func=np.log1p,
            inverse_func=np.expm1,
        )

    else:
        raise ValueError(f"Tarea no soportada: {task}. Debe ser 'classification' o 'regression'")


def create_training_pipeline(
    model_name: str,
    task: str = "classification",
    hyperparameters: Optional[dict[str, Any]] = None,
    use_most_common: bool = False,
) -> ProductionInferenceWrapper:
    """
    Crea el pipeline completo de entrenamiento que incluye el preprocesador
    y el estimador, todo envuelto en el ProductionInferenceWrapper para inferencia.

    Args:
        model_name (str): Nombre del modelo.
        task (str): Tarea.
        hyperparameters (Dict, opcional): Hiperparámetros.
        use_most_common (bool): Usar reducción de cardinalidad en categorías.

    Returns:
        ProductionInferenceWrapper: Pipeline completo listo para producción.
    """
    preprocessor = get_preprocessor_pipeline(use_most_common=use_most_common)
    estimator = build_estimator(
        model_name=model_name,
        task=task,
        hyperparameters=hyperparameters,
    )

    sklearn_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier_or_regressor", estimator),
        ]
    )

    return ProductionInferenceWrapper(pipeline=sklearn_pipeline, task=task)
