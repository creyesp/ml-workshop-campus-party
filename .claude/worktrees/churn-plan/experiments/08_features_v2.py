"""Measure improvement from v2 features and new architectures (spec iteration).

1. Leakage check on the new v2 features (directed univariate AUC; flag >= 0.85).
2. Architecture x feature-set matrix: CV PR-AUC + bootstrap CI for each model on
   the v1 feature set vs the v2 feature set, on the SAME train rows.
3. Tune XGBoost on v2 features and compare its CI to the v1 champion (0.400).

Capped parallelism: every estimator runs single-threaded; folds/models run
sequentially (no oversubscription). Optuna parallelises trials at N_JOBS.

Run: .venv/bin/python experiments/08_features_v2.py [n_trials]
"""
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
from floodit.data.load_v2 import load_v2
from floodit.evaluate.leakage import directed_auc
from floodit.evaluate.metrics import bootstrap_ci, pr_auc
from floodit.features.preprocess import make_preprocessor
from floodit.models.scoring import oof_proba
from floodit.tracking import start_run

SEED = config.RANDOM_SEED
CAT = config.CATEGORICAL_COLUMNS
V1 = config.NUMERICAL_COLUMNS
V2 = config.V2_NUMERICAL_COLUMNS
V1_CHAMPION_PRAUC = 0.400  # tuned XGBoost on v1 (task 12)


def pipe(num_cols, estimator):
    return Pipeline([("pre", make_preprocessor(num_cols, CAT)), ("clf", estimator)])


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
    train = load_v2("train", verify=True)
    y = train[config.LABEL_COLUMN].values

    # --- 1. Leakage check on new v2 features ------------------------------
    new_feats = [c for c in V2 if c not in V1]
    print("=== leakage check on new v2 features (directed AUC) ===")
    leak = pd.DataFrame(
        [{"feature": f, "directed_auc": directed_auc(train, f, y)} for f in new_feats]
    ).sort_values("directed_auc", ascending=False)
    print(leak.to_string(index=False))
    flagged = leak[leak["directed_auc"] >= 0.85]["feature"].tolist()
    print(f"flagged (>=0.85): {flagged or 'none'}")

    # --- 2. Architecture x feature-set matrix -----------------------------
    print("\n=== CV PR-AUC [CI] by architecture x feature set ===")
    print(f"{'model':10s} {'v1 (11 feat)':>22s} {'v2 (22 feat)':>22s}   Δ")
    rows = []
    for name, fac in estimators().items():
        p1, lo1, hi1 = cv_ci(V1, fac, train, y)
        p2, lo2, hi2 = cv_ci(V2, fac, train, y)
        rows.append((name, p1, lo1, hi1, p2, lo2, hi2))
        print(f"{name:10s} {p1:.3f} [{lo1:.3f},{hi1:.3f}]   {p2:.3f} [{lo2:.3f},{hi2:.3f}]  {p2-p1:+.3f}")

    # --- 3. Tune XGBoost on v2 features -----------------------------------
    print(f"\n=== tuning XGBoost on v2 features ({n_trials} trials) ===")

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
        yy, oof = oof_proba(lambda: pipe(V2, est()), train, y, n_splits=5)
        return pr_auc(yy, oof)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, n_jobs=config.N_JOBS)
    best = study.best_params
    est = lambda: XGBClassifier(tree_method="hist", eval_metric="logloss",
                                random_state=SEED, n_jobs=1, **best)
    pt, lo, hi = cv_ci(V2, est, train, y)
    print(f"tuned XGBoost v2  PR-AUC={pt:.3f}  CI=[{lo:.3f}, {hi:.3f}]")
    print(f"v1 champion       PR-AUC={V1_CHAMPION_PRAUC:.3f}  CI=[0.377, 0.424]")
    improved = lo > 0.424
    print("\nVERDICT: " + (
        f"v2 IMPROVES — CI [{lo:.3f},{hi:.3f}] above v1 champion CI [0.377,0.424] (non-overlapping)."
        if improved else
        f"v2 does NOT beat v1 champion with non-overlapping CIs (CI [{lo:.3f},{hi:.3f}] vs [0.377,0.424])."))

    with start_run("v2-xgb-tuned"):
        mlflow.log_params(best)
        mlflow.log_param("feature_set", "v2")
        mlflow.log_metric("pr_auc", pt)
        mlflow.log_metric("pr_auc_ci_low", lo)
        mlflow.log_metric("pr_auc_ci_high", hi)
        mlflow.log_metric("v1_champion_pr_auc", V1_CHAMPION_PRAUC)
        mlflow.set_tag("v2_improves", str(improved))

    import json
    json.dump(best, open("experiments/_artifacts/xgb_v2_best_params.json", "w"), indent=2)


if __name__ == "__main__":
    main()
