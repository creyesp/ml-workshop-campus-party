"""
Orquestador del pipeline de entrenamiento.

Coordina las etapas (lectura -> saneamiento -> split -> entrenamiento -> evaluación
-> registro) delegando toda la lógica de negocio a los módulos especializados. No
contiene split manual, ni ``fit``, ni armado de métricas: solo secuencia llamadas.
"""

from __future__ import annotations

from src.config.settings import get_settings
from src.data.processing import sanitize_dataset, split_data, split_features_target
from src.data.source import load_dataset
from src.train.evaluation import evaluate_model
from src.train.inference import build_inference_model
from src.train.registry import (
    create_run_directory,
    generate_run_id,
    save_metadata,
    save_metrics,
    save_model,
)
from src.train.training import train_model


def run_training(
    model_name: str | None = None,
    family: str | None = None,
    model_params: dict | None = None,
    source_type: str = "csv",
    save: bool = True,
    run_id: str | None = None,
) -> dict:
    """
    Ejecuta el pipeline completo de entrenamiento para un modelo.

    Args:
        model_name: algoritmo a entrenar (default: ``settings.training.default_model``).
        family: familia de salida ``point`` o ``probabilistic``.
        model_params: hiperparámetros que sobrescriben los defaults.
        source_type: fuente de datos (``csv`` / ``bigquery``).
        save: si ``True``, registra artefactos en ``artifacts/<RUN_ID>/``.
        run_id: RUN_ID explícito; si ``None`` se genera por timestamp.

    Returns:
        Dict con ``run_id``, ``metrics``, ``model_name``, ``family`` y el
        ``InferenceModel`` entrenado.
    """
    training_config = get_settings().training
    model_name = model_name or training_config.default_model
    family = family or training_config.default_family
    if family not in training_config.available_families:
        available = training_config.available_families
        raise ValueError(f"Familia desconocida: {family}. Disponibles: {available}")

    # Etapa 1-2: lectura y saneamiento.
    raw = load_dataset(source_type=source_type)
    clean = sanitize_dataset(raw)

    # Etapa 3: split estratificado.
    train_df, test_df = split_data(clean)
    x_train, y_train = split_features_target(train_df)
    x_test, y_test = split_features_target(test_df)

    # Etapa 4-5: entrenamiento (el feature engineering se ajusta dentro del pipeline).
    model = train_model(x_train, y_train, model_name, model_params)

    # Etapa 6: evaluación.
    metrics = evaluate_model(model, x_train, y_train, x_test, y_test)

    # Empaquetado del artefacto de inferencia.
    inference_model = build_inference_model(model, model_name=model_name, family=family)

    result = {
        "run_id": run_id,
        "model_name": model_name,
        "family": family,
        "metrics": metrics,
        "model": inference_model,
    }

    # Etapa 7: registro de artefactos.
    if save:
        result["run_id"] = _persist(inference_model, metrics, result, run_id)

    return result


def _persist(inference_model, metrics: dict, result: dict, run_id: str | None) -> str:
    run_id = run_id or generate_run_id()
    run_dir = create_run_directory(run_id)
    save_model(inference_model, run_dir)
    save_metrics(metrics, run_dir)
    save_metadata(
        {
            "run_id": run_id,
            "model_name": result["model_name"],
            "family": result["family"],
            "decision_threshold": inference_model.threshold,
            "label_column": get_settings().data.label_column,
            "numeric_features": get_settings().data.numeric_features,
            "categorical_features": get_settings().data.categorical_features,
        },
        run_dir,
    )
    if get_settings().verbose:
        print(f"[registry] Artefactos guardados en {run_dir}")
    return run_id
