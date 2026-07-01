"""
Módulo de registro de artefactos (Model Registry).
Maneja el versionado y guardado consistente de modelos, métricas y metadatos
bajo la estructura de carpetas de artifacts/<RUN_ID>/.
"""

import datetime
import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional

import joblib

from src.config.settings import ARTIFACTS_DIR, MODELS_DIR, RUN_ID_FORMAT


def generate_run_id(model_name: str) -> str:
    """
    Genera un identificador único de corrida (RUN_ID) utilizando el timestamp
    actual en formato YYYYMMDDHHMM y el nombre del modelo.

    Args:
        model_name (str): Nombre o identificador del modelo (ej. 'xgb', 'rf').

    Returns:
        str: Identificador único de la corrida.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
    return RUN_ID_FORMAT.format(timestamp=timestamp, model_name=model_name)


def save_run_artifacts(
    run_id: str,
    model: Any,
    metrics_dict: dict[str, Any],
    metadata_dict: dict[str, Any],
    plot_file_path: Optional[str] = None,
) -> Path:
    """
    Guarda de manera estructurada y consistente todos los artefactos generados
    por una corrida de entrenamiento.

    Args:
        run_id (str): Identificador único de corrida.
        model (Any): Pipeline u objeto del modelo entrenado a serializar.
        metrics_dict (Dict[str, Any]): Métricas obtenidas durante la evaluación.
        metadata_dict (Dict[str, Any]): Metadatos de la corrida (parámetros, datos, etc.).
        plot_file_path (str, opcional): Ruta física temporal de los gráficos a reubicar.

    Returns:
        Path: Ruta a la carpeta de la corrida en artifacts/.
    """
    run_dir = ARTIFACTS_DIR / run_id
    os.makedirs(run_dir, exist_ok=True)

    # 1. Guardar el modelo serializado
    model_path = run_dir / "model.joblib"
    joblib.dump(model, model_path)

    # 2. Guardar métricas en formato JSON
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=4, ensure_ascii=False)

    # 3. Guardar metadatos en formato JSON
    metadata_path = run_dir / "metadata.json"
    full_metadata = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        **metadata_dict,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(full_metadata, f, indent=4, ensure_ascii=False)

    # 4. Copiar o reubicar el gráfico de diagnóstico si existe
    if plot_file_path and os.path.exists(plot_file_path):
        target_plot_path = run_dir / "evaluation_plots.png"
        # Si la ruta origen y destino son distintas, mover
        if Path(plot_file_path) != target_plot_path:
            shutil.move(str(plot_file_path), str(target_plot_path))

    # 5. Opcional: registrar una copia del modelo en el directorio /models general
    # para simular un despliegue directo "latest"
    latest_model_path = MODELS_DIR / f"latest_{metadata_dict.get('model_name', 'model')}.joblib"
    joblib.dump(model, latest_model_path)

    return run_dir


def load_model_pipeline(run_id: str) -> Any:
    """
    Carga el pipeline de un modelo guardado en una corrida específica.

    Args:
        run_id (str): Identificador único de corrida.

    Returns:
        Any: Pipeline cargado.
    """
    model_path = ARTIFACTS_DIR / run_id / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo para la corrida {run_id} en {model_path}"
        )

    return joblib.load(model_path)


def load_latest_model(model_type: str) -> Any:
    """
    Carga el modelo marcado como 'latest' para el tipo de modelo especificado
    desde el directorio general de models/.

    Args:
        model_type (str): Nombre del tipo de modelo (ej. 'linear', 'rf', 'xgb').

    Returns:
        Any: Pipeline cargado.
    """
    model_path = MODELS_DIR / f"latest_{model_type}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró un modelo 'latest' para {model_type} en {model_path}"
        )

    return joblib.load(model_path)
