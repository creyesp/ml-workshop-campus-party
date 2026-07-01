import json
from datetime import datetime
from pathlib import Path

import joblib

from src.config.settings import ARTIFACTS_DIR, RUN_ID_FORMAT


def generate_run_id() -> str:
    return datetime.now().strftime(RUN_ID_FORMAT)


def save_artifact(
    pipeline,
    metrics: dict,
    run_id: str,
    model_name: str,
    extra_metadata: dict | None = None,
) -> Path:
    run_dir = ARTIFACTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "model.joblib"
    joblib.dump(pipeline, model_path)
    payload = {
        "run_id": run_id,
        "model_name": model_name,
        "metrics": metrics,
    }
    if extra_metadata:
        payload["metadata"] = extra_metadata
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return run_dir


def load_artifact(run_dir: str | Path) -> tuple:
    run_dir = Path(run_dir)
    pipeline = joblib.load(run_dir / "model.joblib")
    with open(run_dir / "metrics.json") as f:
        metadata = json.load(f)
    return pipeline, metadata


def list_runs() -> list[dict]:
    if not ARTIFACTS_DIR.exists():
        return []
    runs = []
    for entry in sorted(ARTIFACTS_DIR.iterdir()):
        if entry.is_dir():
            metrics_file = entry / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    data = json.load(f)
                runs.append(data)
    return runs
