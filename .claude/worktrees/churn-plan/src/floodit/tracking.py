"""MLflow tracking wrapper (spec §8, datapowers:experiment-tracking).

Every run is auto-tagged with the code SHA and the pinned dataset hashes so a
metric can always be traced back to exact code + data. Uses a local file store
by default (``config.MLRUNS_DIR``); tests pass a temporary ``tracking_uri``.
"""
import subprocess
from contextlib import contextmanager

import mlflow

from floodit import config


def current_git_sha() -> str:
    """Resolve HEAD sha; returns 'unknown' outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=config.REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@contextmanager
def start_run(name: str, experiment: str = "churn", tracking_uri: str | None = None):
    """Start an MLflow run tagged with code SHA + pinned data hashes.

    Parameters
    ----------
    name : run name.
    experiment : MLflow experiment name (created if absent).
    tracking_uri : override store location (defaults to ``config.MLRUNS_DIR``).
    """
    # MLflow 3.x requires a database backend; use a local SQLite store.
    if tracking_uri is None:
        config.MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"sqlite:///{config.MLRUNS_DIR / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tag("git_sha", current_git_sha())
        for split, digest in config.DATA_HASHES.items():
            mlflow.set_tag(f"data_hash_{split}", digest)
        yield run
