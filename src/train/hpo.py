import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score

from src.config.settings import (
    HPO_CV_FOLDS,
    HPO_DEFAULT_METRIC,
    HPO_DEFAULT_N_TRIALS,
    HPO_SEARCH_SPACE,
    RANDOM_STATE,
)
from src.train.training import build_pipeline


def _suggest_params(trial: optuna.Trial, model_name: str) -> dict:
    space = HPO_SEARCH_SPACE[model_name]
    params = {}
    for param_name, spec in space.items():
        param_type = spec["type"]
        if param_type == "float":
            params[param_name] = trial.suggest_float(
                param_name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False),
            )
        elif param_type == "int":
            params[param_name] = trial.suggest_int(
                param_name, spec["low"], spec["high"]
            )
        elif param_type == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, spec["choices"])
    return params


def run_hpo(
    x: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    n_trials: int = HPO_DEFAULT_N_TRIALS,
    metric: str = HPO_DEFAULT_METRIC,
    cv_folds: int = HPO_CV_FOLDS,
    direction: str = "maximize",
) -> optuna.Study:
    def objective(trial):
        params = _suggest_params(trial, model_name)
        pipeline = build_pipeline(model_name=model_name, model_params=params)
        scores = cross_val_score(
            pipeline,
            x,
            y,
            cv=cv_folds,
            scoring=metric,
            error_score="raise",
        )
        return scores.mean()

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials)
    return study
