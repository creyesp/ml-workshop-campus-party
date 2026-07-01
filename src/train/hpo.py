"""
Optimización de hiperparámetros (HPO) desacoplada.

Usa Optuna con validación cruzada interna sobre el conjunto de entrenamiento. El
estudio NO toca el test final: la selección de hiperparámetros se valida únicamente
con folds de train, evitando fuga hacia la evaluación. El espacio de búsqueda y el
número de trials provienen de la configuración.
"""

from __future__ import annotations

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config.settings import get_settings
from src.data.processing import (
    sanitize_dataset,
    split_data,
    split_features_target,
)
from src.data.source import load_dataset
from src.train.training import build_model


def _suggest(trial: optuna.Trial, name: str, spec: dict):
    """Traduce una entrada del espacio de búsqueda a una sugerencia de Optuna."""
    kind = spec["type"]
    if kind == "int":
        return trial.suggest_int(
            name, spec["low"], spec["high"], step=spec.get("step", 1)
        )
    if kind == "float":
        return trial.suggest_float(
            name, spec["low"], spec["high"], log=spec.get("log", False)
        )
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Tipo de parámetro no soportado: {kind}")


def run_hpo(model_name: str | None = None, n_trials: int | None = None) -> dict:
    """
    Ejecuta el estudio de HPO para un algoritmo.

    Args:
        model_name: algoritmo a optimizar (debe tener espacio de búsqueda definido).
        n_trials: número de trials; por defecto el de la configuración.

    Returns:
        Dict con ``model_name``, ``best_params``, ``best_score`` y ``n_trials``.
    """
    settings = get_settings()
    hpo_config = settings.hpo
    model_name = model_name or settings.training.default_model
    n_trials = n_trials or hpo_config.n_trials

    search_space = hpo_config.search_spaces.get(model_name)
    if not search_space:
        raise ValueError(
            f"No hay espacio de búsqueda definido para '{model_name}'. "
            f"Disponibles: {list(hpo_config.search_spaces)}"
        )

    # Solo se usa train para el estudio; el test permanece intacto.
    clean = sanitize_dataset(load_dataset())
    train_df, _test_df = split_data(clean)
    x_train, y_train = split_features_target(train_df)

    cv = StratifiedKFold(
        n_splits=hpo_config.cv_folds,
        shuffle=True,
        random_state=settings.random_seed,
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            name: _suggest(trial, name, spec) for name, spec in search_space.items()
        }
        model = build_model(model_name, y_train=y_train, **params)
        scores = cross_val_score(
            model, x_train, y_train, cv=cv, scoring=hpo_config.metric_name
        )
        return float(np.mean(scores))

    study = optuna.create_study(direction=hpo_config.direction)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    result = {
        "model_name": model_name,
        "metric": hpo_config.metric_name,
        "best_params": study.best_params,
        "best_score": float(study.best_value),
        "n_trials": len(study.trials),
    }
    if settings.verbose:
        print(
            f"[hpo] {model_name}: mejor {hpo_config.metric_name}="
            f"{result['best_score']:.4f} con {result['best_params']}"
        )
    return result


def main(argv: list[str] | None = None) -> None:
    """CLI mínimo del estudio de HPO: ``python -m src.train.hpo``."""
    import argparse
    import json

    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="src.train.hpo", description="HPO con Optuna."
    )
    parser.add_argument(
        "--model",
        choices=settings.training.available_models,
        default=settings.training.default_model,
    )
    parser.add_argument("--trials", type=int, help="Número de trials.")
    args = parser.parse_args(argv)
    result = run_hpo(model_name=args.model, n_trials=args.trials)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
