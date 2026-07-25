"""Tuned XGBoost candidate (spec §7 H2).

``build_xgb`` wraps the standard preprocessor + an XGBClassifier. ``tune_xgb``
runs an Optuna study maximising 5-fold out-of-fold PR-AUC.

Parallelism (user constraint — never use all cores): Optuna parallelises trials
at ``N_JOBS`` while each in-trial XGBoost runs single-threaded, so total CPU use
is bounded by ``N_JOBS`` (no nested oversubscription). The final model built for
scoring uses ``n_jobs=N_JOBS``.
"""
import optuna
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from floodit import config
from floodit.evaluate.metrics import pr_auc
from floodit.features.preprocess import build_preprocessor
from floodit.models.scoring import oof_proba

optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_xgb(params: dict, n_jobs: int = config.N_JOBS) -> Pipeline:
    clf = XGBClassifier(
        random_state=config.RANDOM_SEED,
        n_jobs=n_jobs,
        eval_metric="logloss",
        tree_method="hist",
        **params,
    )
    return Pipeline(steps=[("pre", build_preprocessor()), ("clf", clf)])


def _suggest(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        # Counter the ~23% positive class; tune around the natural ratio (~3.3).
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 6.0),
    }


def tune_xgb(df, y, n_trials: int = 30, seed: int = config.RANDOM_SEED):
    """Return ``(best_params, best_value, study)`` maximising CV PR-AUC."""

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial)
        # single-threaded xgb inside trials; Optuna provides the parallelism
        factory = lambda: build_xgb(params, n_jobs=1)
        yy, oof = oof_proba(factory, df, y, n_splits=5)
        return pr_auc(yy, oof)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, n_jobs=config.N_JOBS)
    return study.best_params, study.best_value, study
