"""
Módulo de ajuste de hiperparámetros (HPO).
Implementa optimización Bayesiana con Optuna mediante validación cruzada
sobre el conjunto de entrenamiento, aislando el conjunto de test.
"""

from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from src.config.settings import (
    DEFAULT_RANDOM_STATE,
    HPO_N_SPLITS,
    HPO_NUM_TRIALS,
    HPO_TIMEOUT_SECONDS,
    LABEL_COLUMN,
)
from src.features.build import get_preprocessor_pipeline
from src.train.training import build_estimator


def run_hpo_study(
    train_df: pd.DataFrame,
    model_name: str,
    task: str = "classification",
    n_trials: int = HPO_NUM_TRIALS,
    timeout: int = HPO_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """
    Ejecuta un estudio de Optuna para encontrar los mejores hiperparámetros.

    Args:
        train_df (pd.DataFrame): Datos de entrenamiento.
        model_name (str): Nombre del modelo ('xgb', 'rf', etc.).
        task (str): Tarea ('classification' o 'regression').
        n_trials (int): Número de trials a ejecutar.
        timeout (int): Tiempo límite en segundos.

    Returns:
        Dict[str, Any]: Diccionario con los mejores parámetros encontrados.
    """
    # Desactivar logs del estudio Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    x_train = train_df.drop(columns=[LABEL_COLUMN], errors="ignore")
    y_train = train_df[LABEL_COLUMN]

    # Preprocesar datos de forma anticipada para acelerar los trials
    preprocessor = get_preprocessor_pipeline(use_most_common=False)
    x_train_transformed = preprocessor.fit_transform(x_train, y_train)

    def objective(trial: optuna.Trial) -> float:
        # 1. Sugerir hiperparámetros según el modelo
        params = {}
        if model_name == "xgb":
            params = {
                "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
                "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            }
            if task == "classification":
                # Calcular peso de balanceo si es clasificación
                class_counts = y_train.value_counts()
                if len(class_counts) > 1 and 0 in class_counts:
                    params["scale_pos_weight"] = float(class_counts[0] / class_counts[1])
                params["eval_metric"] = "logloss"

        elif model_name == "rf":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 15, log=False),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            }

        elif model_name == "linear":
            if task == "classification":
                params = {
                    "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
                    "penalty": trial.suggest_categorical("penalty", ["l2"]),
                }
            else:
                params = {
                    "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
                }

        else:
            return 0.0

        # 2. Configurar validación cruzada interna (evitar fugas hacia el test)
        if task == "classification":
            kf = StratifiedKFold(
                n_splits=HPO_N_SPLITS,
                shuffle=True,
                random_state=DEFAULT_RANDOM_STATE,
            )
        else:
            kf = KFold(
                n_splits=HPO_N_SPLITS,
                shuffle=True,
                random_state=DEFAULT_RANDOM_STATE,
            )

        scores = []
        for train_idx, val_idx in kf.split(x_train_transformed, y_train):
            # Dividir pliegues en numpy array ya preprocesado
            x_tr, x_val = x_train_transformed[train_idx], x_train_transformed[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Construir y entrenar estimador
            estimator = build_estimator(
                model_name=model_name,
                task=task,
                hyperparameters=params,
            )
            estimator.fit(x_tr, y_tr)

            # Predecir y calcular score según la tarea
            if task == "classification":
                if hasattr(estimator, "predict_proba"):
                    y_prob = estimator.predict_proba(x_val)[:, 1]
                else:
                    y_prob = estimator.predict(x_val)
                score = roc_auc_score(y_val, y_prob)
            else:
                y_pred = estimator.predict(x_val)
                score = -float(np.sqrt(mean_squared_error(y_val, y_pred)))

            scores.append(score)

        return float(np.mean(scores))

    # Definir dirección del estudio
    direction = "maximize"
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    return {
        "best_params": study.best_value,
        "best_hyperparameters": study.best_params,
        "n_trials_completed": len(study.trials),
    }
