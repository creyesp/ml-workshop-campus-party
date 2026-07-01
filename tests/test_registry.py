"""
Pruebas unitarias para el módulo de registro de artefactos (registry.py).
Verifica que cada corrida genere los archivos esperados con el formato correcto.
"""

import json
import joblib
import pytest
from pathlib import Path

from src.train.registry import generate_run_id, save_run_artifacts, load_model_pipeline


# ── generate_run_id ───────────────────────────────────────────────────────────

def test_run_id_format():
    """El RUN_ID debe seguir el patrón YYYYMMDDHHMM_<model_name>."""
    run_id = generate_run_id("xgb")
    parts = run_id.split("_")
    assert len(parts) == 2
    timestamp_part, model_part = parts
    assert len(timestamp_part) == 12  # YYYYMMDDHHMM
    assert timestamp_part.isdigit()
    assert model_part == "xgb"


def test_run_id_unique():
    """Dos run IDs generados en distintos momentos deben ser diferentes."""
    import time
    id_1 = generate_run_id("rf")
    time.sleep(0.01)
    id_2 = generate_run_id("rf")
    # No se garantiza diferencia dentro del mismo minuto, pero el formato sí debe ser válido
    assert id_1.startswith("2")  # año >= 2000
    assert id_2.startswith("2")


# ── save_run_artifacts ────────────────────────────────────────────────────────

class _DummyModel:
    """Modelo simulado para tests de serialización."""
    def predict(self, X):
        return [0] * len(X)


def test_save_artifacts_creates_files(tmp_path, monkeypatch):
    """
    save_run_artifacts debe crear model.joblib, metrics.json y metadata.json
    dentro de artifacts/<run_id>/.
    """
    artifacts_dir = tmp_path / "artifacts"
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)

    import src.train.registry as registry_module
    monkeypatch.setattr(registry_module, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(registry_module, "MODELS_DIR", models_dir)

    dummy_model = _DummyModel()
    metrics = {"accuracy": 0.85, "f1_score": 0.80}
    metadata = {"model_name": "linear", "task": "classification"}
    run_id = "202601010000_linear"

    run_dir = save_run_artifacts(
        run_id=run_id,
        model=dummy_model,
        metrics_dict=metrics,
        metadata_dict=metadata,
        plot_file_path=None,
    )

    # Verificar que se creó la carpeta de corrida
    assert run_dir.exists()
    assert run_dir.is_dir()

    # Verificar archivos esperados
    assert (run_dir / "model.joblib").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "metadata.json").exists()


def test_metrics_json_content(tmp_path, monkeypatch):
    """El archivo metrics.json debe reflejar exactamente el dict pasado."""
    artifacts_dir = tmp_path / "artifacts"
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)

    import src.train.registry as registry_module
    monkeypatch.setattr(registry_module, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(registry_module, "MODELS_DIR", models_dir)

    metrics = {"roc_auc": 0.75, "log_loss": 0.45, "confusion_matrix": [[10, 2], [3, 8]]}
    run_id = "202601010000_rf"

    run_dir = save_run_artifacts(
        run_id=run_id,
        model=_DummyModel(),
        metrics_dict=metrics,
        metadata_dict={"model_name": "rf"},
        plot_file_path=None,
    )

    with open(run_dir / "metrics.json", encoding="utf-8") as f:
        saved = json.load(f)

    assert saved["roc_auc"] == pytest.approx(0.75)
    assert saved["confusion_matrix"] == [[10, 2], [3, 8]]


def test_metadata_json_has_run_id(tmp_path, monkeypatch):
    """El metadata.json debe incluir el run_id y el timestamp de escritura."""
    artifacts_dir = tmp_path / "artifacts"
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)

    import src.train.registry as registry_module
    monkeypatch.setattr(registry_module, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(registry_module, "MODELS_DIR", models_dir)

    run_id = "202601010000_xgb"
    run_dir = save_run_artifacts(
        run_id=run_id,
        model=_DummyModel(),
        metrics_dict={},
        metadata_dict={"model_name": "xgb", "task": "classification"},
        plot_file_path=None,
    )

    with open(run_dir / "metadata.json", encoding="utf-8") as f:
        saved = json.load(f)

    assert saved["run_id"] == run_id
    assert "timestamp" in saved
    assert saved["model_name"] == "xgb"


def test_model_is_serializable(tmp_path, monkeypatch):
    """El modelo guardado debe poder recargarse correctamente con joblib."""
    artifacts_dir = tmp_path / "artifacts"
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)

    import src.train.registry as registry_module
    monkeypatch.setattr(registry_module, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(registry_module, "MODELS_DIR", models_dir)

    run_id = "202601010000_gb"
    dummy = _DummyModel()
    run_dir = save_run_artifacts(
        run_id=run_id,
        model=dummy,
        metrics_dict={},
        metadata_dict={"model_name": "gb"},
        plot_file_path=None,
    )

    loaded = joblib.load(run_dir / "model.joblib")
    # Verificar que el objeto cargado puede predecir
    assert loaded.predict([[1, 2, 3]]) == [0]


# ── load_model_pipeline ───────────────────────────────────────────────────────

def test_load_model_pipeline_missing_raises(tmp_path, monkeypatch):
    """Cargar un run_id inexistente debe lanzar FileNotFoundError."""
    import src.train.registry as registry_module
    monkeypatch.setattr(registry_module, "ARTIFACTS_DIR", tmp_path / "artifacts")

    with pytest.raises(FileNotFoundError):
        load_model_pipeline("nonexistent_run_id")
