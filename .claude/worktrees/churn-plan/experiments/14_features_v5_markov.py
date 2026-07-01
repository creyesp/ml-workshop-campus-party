"""Markov sequence-ORDER features v5 vs. v3 champion + architecture sweep.

Open question after v4: raw navigation COUNTS added nothing. Does the transition
ORDER (first-order Markov transition PROBABILITIES P(cur|prev) + a few 2-gram
flags) carry orthogonal churn signal? This is a cheap falsification test BEFORE
considering a sequence DNN.

1. Leakage check on the new Markov features (directed AUC; flag >= 0.85).
2. Architecture x feature-set matrix: CV PR-AUC + CI for each model on v3 (32
   feat) vs v5 (v3 + 19 markov = 51 feat), same train rows.
3. Tune XGBoost on v5 (Optuna) and compare its CI to the v3 champion
   (0.548 [0.520, 0.575]).

DECISION RULE: v5 wins ONLY if its tuned CI is strictly above the v3 champion CI
(non-overlapping, lo > 0.575). Otherwise the Markov/order hypothesis is REJECTED.

Capped parallelism: estimators single-threaded; Optuna trials at N_JOBS.

Run: .venv/bin/python experiments/14_features_v5_markov.py [n_trials]
"""
import json
import sys

import mlflow
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from floodit import config
from floodit.data.load_v5 import load_v5
from floodit.evaluate.leakage import directed_auc
from floodit.evaluate.metrics import bootstrap_ci, pr_auc
from floodit.features.preprocess import make_preprocessor
from floodit.models.scoring import oof_proba
from floodit.tracking import start_run

SEED = config.RANDOM_SEED
CAT = config.CATEGORICAL_COLUMNS
V3 = config.V3_NUMERICAL_COLUMNS
V5 = config.V5_NUMERICAL_COLUMNS
V3_CHAMP_LO, V3_CHAMP_HI = 0.520, 0.575  # tuned XGBoost on v3 (task 19, champion)


def pipe(num_cols, est):
    return Pipeline([("pre", make_preprocessor(num_cols, CAT)), ("clf", est)])


def estimators():
    return {
        "xgboost": lambda: XGBClassifier(
            n_estimators=300, max_depth=2, learning_rate=0.069, subsample=0.87,
            colsample_bytree=0.91, min_child_weight=3, reg_lambda=4.43,
            scale_pos_weight=4.12, tree_method="hist", eval_metric="logloss",
            random_state=SEED, n_jobs=1),
        "lightgbm": lambda: LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=15,
            class_weight="balanced", random_state=SEED, n_jobs=1, verbose=-1),
        "histgb": lambda: HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300,
            class_weight="balanced", random_state=SEED),
        "catboost": lambda: CatBoostClassifier(
            iterations=300, depth=3, learning_rate=0.05,
            auto_class_weights="Balanced", random_seed=SEED,
            thread_count=1, verbose=0),
    }


def cv_ci(num_cols, est_factory, df, y):
    yy, oof = oof_proba(lambda: pipe(num_cols, est_factory()), df, y, n_splits=5)
    lo, mid, hi = bootstrap_ci(yy, oof, metric=pr_auc, n=1000)
    return pr_auc(yy, oof), lo, hi


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    train = load_v5("train", verify=True)
    y = train[config.LABEL_COLUMN].values

    new_feats = config.V5_MARKOV_COLUMNS
    print("=== leakage check on new Markov features (directed AUC) ===")
    leak = pd.DataFrame(
        [{"feature": f, "directed_auc": directed_auc(train, f, y)} for f in new_feats]
    ).sort_values("directed_auc", ascending=False)
    print(leak.to_string(index=False))
    flagged = leak[leak["directed_auc"] >= 0.85]["feature"].tolist()
    print(f"flagged (>=0.85): {flagged or 'none'}")

    print("\n=== CV PR-AUC [CI] by architecture x feature set ===")
    print(f"{'model':10s} {'v3 (32 feat)':>22s} {'v5 (51 feat)':>22s}   Δ")
    for name, fac in estimators().items():
        p3, lo3, hi3 = cv_ci(V3, fac, train, y)
        p5, lo5, hi5 = cv_ci(V5, fac, train, y)
        print(f"{name:10s} {p3:.3f} [{lo3:.3f},{hi3:.3f}]   {p5:.3f} [{lo5:.3f},{hi5:.3f}]  {p5-p3:+.3f}")

    print(f"\n=== tuning XGBoost on v5 features ({n_trials} trials) ===")

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            scale_pos_weight=trial.suggest_float("scale_pos_weight", 1.0, 6.0),
        )
        est = lambda: XGBClassifier(tree_method="hist", eval_metric="logloss",
                                    random_state=SEED, n_jobs=1, **params)
        yy, oof = oof_proba(lambda: pipe(V5, est()), train, y, n_splits=5)
        return pr_auc(yy, oof)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, n_jobs=config.N_JOBS)
    best = study.best_params
    est = lambda: XGBClassifier(tree_method="hist", eval_metric="logloss",
                                random_state=SEED, n_jobs=1, **best)
    pt, lo, hi = cv_ci(V5, est, train, y)
    print(f"tuned XGBoost v5  PR-AUC={pt:.3f}  CI=[{lo:.3f}, {hi:.3f}]")
    print(f"v3 champion       PR-AUC=0.548  CI=[{V3_CHAMP_LO}, {V3_CHAMP_HI}]")
    improved = lo > V3_CHAMP_HI
    print("\nVERDICT: " + (
        f"v5 IMPROVES (Markov/order SUPPORTED) — CI [{lo:.3f},{hi:.3f}] strictly above "
        f"v3 champion CI [{V3_CHAMP_LO},{V3_CHAMP_HI}] (non-overlapping)."
        if improved else
        f"Markov/order REJECTED — v5 tuned CI [{lo:.3f},{hi:.3f}] not strictly above "
        f"v3 champion CI [{V3_CHAMP_LO},{V3_CHAMP_HI}] (overlapping)."))

    with start_run("v5-xgb-tuned"):
        mlflow.log_params(best)
        mlflow.log_param("feature_set", "v5")
        mlflow.log_metric("pr_auc", pt)
        mlflow.log_metric("pr_auc_ci_low", lo)
        mlflow.log_metric("pr_auc_ci_high", hi)
        mlflow.set_tag("v5_improves", str(improved))
    json.dump(best, open("experiments/_artifacts/xgb_v5_best_params.json", "w"), indent=2)


if __name__ == "__main__":
    main()
