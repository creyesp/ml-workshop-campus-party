"""
Artefacto de inferencia.

``InferenceModel`` empaqueta en un único objeto serializable el pipeline entrenado
(features + clasificador), el umbral de decisión y la familia de salida. Su método
``predict`` devuelve directamente salida de negocio (probabilidad de churn y/o clase),
sin postprocesamiento manual fuera del modelo.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config.settings import get_settings
from src.train.training import predict_label, predict_proba


@dataclass
class InferenceModel:
    """Modelo listo para inferencia con salida de negocio."""

    pipeline: Pipeline
    model_name: str
    family: str
    threshold: float

    def predict(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Devuelve la salida de negocio para cada registro.

        - ``churn_probability``: probabilidad estimada de churn.
        - ``churn_prediction``: clase (0/1) según el umbral. Solo se incluye en la
          familia ``point`` (donde la decisión de clase es la salida esperada), pero
          la probabilidad siempre está disponible.
        """
        probability = predict_proba(self.pipeline, dataframe)
        output = pd.DataFrame({"churn_probability": probability}, index=dataframe.index)
        if self.family == "point":
            output["churn_prediction"] = predict_label(
                self.pipeline, dataframe, threshold=self.threshold
            )
        return output

    def save(self, path) -> None:
        """Serializa el artefacto completo en disco."""
        joblib.dump(self, path)


def load_model(path) -> InferenceModel:
    """Carga un ``InferenceModel`` serializado."""
    return joblib.load(path)


def build_inference_model(
    pipeline: Pipeline,
    model_name: str,
    family: str,
    threshold: float | None = None,
) -> InferenceModel:
    """Crea el artefacto de inferencia a partir de un pipeline entrenado."""
    if threshold is None:
        threshold = get_settings().training.decision_threshold
    return InferenceModel(
        pipeline=pipeline,
        model_name=model_name,
        family=family,
        threshold=threshold,
    )
