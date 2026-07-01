"""
Registro de artefactos por corrida.

Cada ejecución escribe en ``artifacts/<RUN_ID>/`` donde ``RUN_ID`` es un timestamp
con el formato definido en la configuración (``RUN_ID_FORMAT``). Se guardan de forma
consistente el modelo de inferencia, las métricas y los metadatos.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.config.settings import get_settings
from src.train.inference import InferenceModel


def generate_run_id(now: datetime | None = None) -> str:
    """Genera un RUN_ID a partir del timestamp actual y el formato configurado."""
    stamp = now or datetime.now()
    return stamp.strftime(get_settings().artifacts.run_id_format)


def create_run_directory(run_id: str) -> Path:
    """Crea (si no existe) el directorio de la corrida y lo devuelve."""
    run_dir = get_settings().artifacts.artifacts_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_model(model: InferenceModel, run_dir: Path) -> Path:
    """Guarda el artefacto de inferencia."""
    path = run_dir / get_settings().artifacts.model_filename
    model.save(path)
    return path


def _dump_json(payload: dict, path: Path) -> Path:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


def save_metrics(metrics: dict, run_dir: Path) -> Path:
    """Guarda las métricas de train/test."""
    return _dump_json(metrics, run_dir / get_settings().artifacts.metrics_filename)


def save_metadata(metadata: dict, run_dir: Path) -> Path:
    """Guarda los metadatos de la corrida."""
    return _dump_json(metadata, run_dir / get_settings().artifacts.metadata_filename)


def save_comparison(comparison: dict, run_dir: Path) -> Path:
    """Guarda la salida estructurada de la comparación de modelos."""
    return _dump_json(
        comparison, run_dir / get_settings().artifacts.comparison_filename
    )
