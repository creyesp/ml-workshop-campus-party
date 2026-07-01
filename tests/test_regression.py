"""Test de regresión sobre el artefacto congelado ``artifacts/model_v1``.

Protege el modelo campeón: si un refactor cambia el pipeline o el preprocesamiento
y las métricas se desvían, este test falla. Se salta si el artefacto aún no fue
generado (``floodit-train``).
"""
from __future__ import annotations

import pytest

from floodit_churn.config import ARTIFACTS_DIR, TEST_CSV
from floodit_churn.data import load_dataset
from floodit_churn.model import evaluate, load_metadata, load_model

MODEL_DIR = ARTIFACTS_DIR / "model_v1"

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "model.joblib").exists(),
    reason="artifacts/model_v1 no generado (corre `uv run floodit-train`)",
)


def test_frozen_model_reproduces_metrics():
    model = load_model(MODEL_DIR)
    meta = load_metadata(MODEL_DIR)
    df = load_dataset(TEST_CSV, require_label=True)
    current = evaluate(model, df, threshold=meta.get("default_threshold", 0.5))
    expected = meta["test_metrics"]
    # PR-AUC y ROC-AUC deben reproducirse dentro de tolerancia estrecha.
    assert current["pr_auc"] == pytest.approx(expected["pr_auc"], abs=1e-6)
    assert current["roc_auc"] == pytest.approx(expected["roc_auc"], abs=1e-6)
