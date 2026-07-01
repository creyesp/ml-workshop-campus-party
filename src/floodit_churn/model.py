"""Definición, entrenamiento y persistencia del modelo XGBoost ganador.

El modelo ganador del workshop (notebooks ``3.3_modeling_xgb`` / ``5.0_explanability``)
es un ``Pipeline`` = preprocessor + ``XGBClassifier`` balanceado por ``scale_pos_weight``.

Nota sobre el tuneo: el notebook ``4.0_hp_xgboost`` corre Optuna, pero ese search
NO es reproducible (estudio sin semilla, sólo 5 trials, y ``best_parameters`` se
sobrescribe con ``study.trials[1].params``). Por eso se congela como v1 el XGBoost
base determinista (``random_state=42``). Los hiperparámetros quedan expuestos en
``HYPERPARAMS`` para ajustar/re-tunear en el futuro sin tocar el resto del código.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import sklearn
import xgboost
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import __version__
from .config import FEATURE_COLUMNS, RANDOM_STATE
from .data import sha256, split_xy
from .features import build_preprocessor
from .metrics import classification_metrics

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"

# Hiperparámetros congelados del XGBoost ganador (ajustables para re-tunear).
# ``scale_pos_weight`` se deriva de los datos en tiempo de entrenamiento.
HYPERPARAMS: dict[str, Any] = {
    "random_state": RANDOM_STATE,
    "eval_metric": "logloss",
    "objective": "binary:logistic",
}


def build_pipeline(scale_pos_weight: float, hyperparams: dict[str, Any] | None = None) -> Pipeline:
    """Construye el pipeline preprocessor + XGBClassifier ganador."""
    params = {**HYPERPARAMS, **(hyperparams or {})}
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", XGBClassifier(scale_pos_weight=scale_pos_weight, **params)),
        ]
    )


def train(df, hyperparams: dict[str, Any] | None = None) -> Pipeline:
    """Entrena el pipeline sobre un DataFrame con label ``churned``."""
    x, y = split_xy(df)
    scale_pos_weight = float(1.0 / y.mean())
    model = build_pipeline(scale_pos_weight, hyperparams)
    model.fit(x, y)
    return model


def evaluate(model: Pipeline, df, threshold: float = 0.5) -> dict[str, float]:
    """Evalúa el modelo sobre un DataFrame etiquetado."""
    x, y = split_xy(df)
    proba = model.predict_proba(x)[:, 1]
    return classification_metrics(y, proba, threshold=threshold)


def save_model(
    model: Pipeline,
    out_dir: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persiste el modelo y un ``metadata.json`` con info de reproducibilidad."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / MODEL_FILENAME)

    meta: dict[str, Any] = {
        "package_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "feature_columns": FEATURE_COLUMNS,
        "hyperparams": HYPERPARAMS,
    }
    meta.update(metadata or {})
    with open(out_dir / METADATA_FILENAME, "w") as f:
        json.dump(meta, f, indent=2)
    return out_dir


def load_model(model_dir: str | Path) -> Pipeline:
    """Carga un modelo persistido desde su directorio de artefacto."""
    return joblib.load(Path(model_dir) / MODEL_FILENAME)


def load_metadata(model_dir: str | Path) -> dict[str, Any]:
    """Carga el ``metadata.json`` del artefacto (o dict vacío si no existe)."""
    path = Path(model_dir) / METADATA_FILENAME
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def train_and_freeze(
    train_csv: str | Path,
    out_dir: str | Path,
    test_csv: str | Path | None = None,
    threshold: float = 0.5,
) -> Path:
    """Entrena sobre ``train_csv``, evalúa (si hay test) y congela el artefacto.

    Registra en la metadata el hash de los datos y las métricas de evaluación.
    """
    from .data import load_dataset

    train_df = load_dataset(train_csv, require_label=True)
    model = train(train_df)

    metadata: dict[str, Any] = {
        "default_threshold": threshold,
        "train_csv": str(train_csv),
        "train_sha256": sha256(train_csv),
        "n_train": int(len(train_df)),
    }
    if test_csv is not None:
        test_df = load_dataset(test_csv, require_label=True)
        metadata["test_csv"] = str(test_csv)
        metadata["test_sha256"] = sha256(test_csv)
        metadata["n_test"] = int(len(test_df))
        metadata["test_metrics"] = evaluate(model, test_df, threshold=threshold)

    return save_model(model, out_dir, metadata)
