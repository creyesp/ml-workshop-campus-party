"""Navigation-sequence features v4 vs. v3 + architecture sweep (spec iteration 3).

1. Leakage check on the new navigation features (directed AUC; flag >= 0.85).
2. Architecture x feature-set matrix: CV PR-AUC + CI for each model on v3 (32
   feat) vs v4 (47 feat), same train rows.
3. Tune XGBoost on v4 and compare its CI to the v3 champion (0.548 [0.520,0.575]).

Capped parallelism: estimators single-threaded; Optuna trials at N_JOBS.

Run: .venv/bin/python experiments/12_features_v4.py [n_trials]
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
from floodit.data.load_v4 import load_v4
from floodit.evaluate.leakage import directed_auc
from floodit.evaluate.metrics import bootstrap_ci, pr_auc
from floodit.features.preprocess import make_preprocessor
from floodit.models.scoring import oof_proba
from floodit.tracking import start_run

SEED = config.RANDOM_SEED
CAT = config.CATEGORICAL_COLUMNS
V2 = config.V3_NUMERICAL_COLUMNS
V3 = config.V4_NUMERICAL_COLUMNS
V2_CHAMP_LO, V2_CHAMP_HI = 0.520, 0.575  # tuned XGBoost on v3 (task 19)


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
    train = load_v4("train", verify=True)
    y = train[config.LABEL_COLUMN].values

    new_feats = [c for c in V3 if c not in V2]
    print("=== leakage check on new navigation features (directed AUC) ===")
    leak = pd.DataFrame(
        [{"feature": f, "directed_auc": directed_auc(train, f, y)} for f in new_feats]
    ).sort_values("directed_auc", ascending=False)
    print(leak.to_string(index=False))
    flagged = leak[leak["directed_auc"] >= 0.85]["feature"].tolist()
    print(f"flagged (>=0.85): {flagged or 'none'}")

    print("\n=== CV PR-AUC [CI] by architecture x feature set ===")
    print(f"{'model':10s} {'v3 (32 feat)':>22s} {'v4 (47 feat)':>22s}   Δ")
    for name, fac in estimators().items():
        p2, lo2, hi2 = cv_ci(V2, fac, train, y)
        p3, lo3, hi3 = cv_ci(V3, fac, train, y)
        print(f"{name:10s} {p2:.3f} [{lo2:.3f},{hi2:.3f}]   {p3:.3f} [{lo3:.3f},{hi3:.3f}]  {p3-p2:+.3f}")

    print(f"\n=== tuning XGBoost on v4 features ({n_trials} trials) ===")

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
        yy, oof = oof_proba(lambda: pipe(V3, est()), train, y, n_splits=5)
        return pr_auc(yy, oof)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, n_jobs=config.N_JOBS)
    best = study.best_params
    est = lambda: XGBClassifier(tree_method="hist", eval_metric="logloss",
                                random_state=SEED, n_jobs=1, **best)
    pt, lo, hi = cv_ci(V3, est, train, y)
    print(f"tuned XGBoost v4  PR-AUC={pt:.3f}  CI=[{lo:.3f}, {hi:.3f}]")
    print(f"v3 champion       PR-AUC=0.548  CI=[{V2_CHAMP_LO}, {V2_CHAMP_HI}]")
    improved = lo > V2_CHAMP_HI
    print("\nVERDICT: " + (
        f"v4 IMPROVES — CI [{lo:.3f},{hi:.3f}] above v3 champion CI [{V2_CHAMP_LO},{V2_CHAMP_HI}] (non-overlapping)."
        if improved else
        f"v4 does NOT beat v2 with non-overlapping CIs (CI [{lo:.3f},{hi:.3f}] vs [{V2_CHAMP_LO},{V2_CHAMP_HI}])."))

    with start_run("v4-xgb-tuned"):
        mlflow.log_params(best)
        mlflow.log_param("feature_set", "v4")
        mlflow.log_metric("pr_auc", pt)
        mlflow.log_metric("pr_auc_ci_low", lo)
        mlflow.log_metric("pr_auc_ci_high", hi)
        mlflow.set_tag("v4_improves", str(improved))
    json.dump(best, open("experiments/_artifacts/xgb_v4_best_params.json", "w"), indent=2)


if __name__ == "__main__":
    main()
