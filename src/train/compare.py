"""
Comparación entre familias/algoritmos de modelos.

Entrena cada modelo solicitado, recolecta sus métricas de test y produce un ranking
por ROC-AUC. La salida estructurada se persiste en ``comparison.json`` dentro del
directorio de la corrida de comparación.
"""

from __future__ import annotations

from src.config.settings import get_settings
from src.train.pipeline import run_training
from src.train.registry import create_run_directory, generate_run_id, save_comparison


def compare_models(
    model_names: list[str] | None = None,
    family: str | None = None,
    save: bool = True,
) -> dict:
    """
    Entrena y compara varios modelos sobre el mismo split.

    Args:
        model_names: lista de algoritmos; por defecto todos los disponibles.
        family: familia de salida usada para todos.
        save: si ``True``, guarda ``comparison.json`` y cada modelo entrenado.

    Returns:
        Dict con ``ranking`` (ordenado por ROC-AUC de test desc) y ``results``.
    """
    settings = get_settings()
    model_names = model_names or settings.training.available_models

    run_id = generate_run_id()
    results: list[dict] = []
    for name in model_names:
        outcome = run_training(
            model_name=name,
            family=family,
            save=save,
            run_id=run_id,
        )
        test_metrics = outcome["metrics"]["test"]
        results.append(
            {
                "model_name": name,
                "roc_auc": test_metrics["roc_auc"],
                "average_precision": test_metrics["average_precision"],
                "f1": test_metrics["f1"],
                "recall": test_metrics["recall"],
                "precision": test_metrics["precision"],
            }
        )

    ranking = sorted(results, key=lambda row: row["roc_auc"], reverse=True)
    comparison = {
        "run_id": run_id,
        "metric": "roc_auc",
        "best_model": ranking[0]["model_name"] if ranking else None,
        "ranking": ranking,
    }

    if save:
        run_dir = create_run_directory(run_id)
        save_comparison(comparison, run_dir)
        if settings.verbose:
            print(f"[compare] comparison.json guardado en {run_dir}")

    return comparison
