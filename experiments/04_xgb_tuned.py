"""H2 — Optuna-tuned XGBoost vs. the strong LogReg baseline (spec §7 H2, §5).

Tunes XGBoost by 5-fold OOF PR-AUC, then scores the best config with the same
OOF protocol and bootstraps a PR-AUC CI. H2 holds iff the XGBoost CI does NOT
overlap the LogReg baseline CI; otherwise keep the simpler model.

Run: .venv/bin/python experiments/04_xgb_tuned.py [n_trials]
"""
import sys

import mlflow

from floodit import config
from floodit.data.contract import validate_contract
from floodit.data.load import load_dataset
from floodit.evaluate.metrics import bootstrap_ci, pr_auc, roc_auc
from floodit.models.baseline import logreg_baseline
from floodit.models.scoring import oof_proba
from floodit.models.xgb import build_xgb, tune_xgb
from floodit.tracking import start_run


def ci(factory, df, y):
    yy, oof = oof_proba(factory, df, y, n_splits=5)
    lo, mid, hi = bootstrap_ci(yy, oof, metric=pr_auc, n=1000)
    return pr_auc(yy, oof), lo, hi, roc_auc(yy, oof)


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    df = load_dataset("train", verify=True)
    validate_contract(df)
    y = df[config.LABEL_COLUMN].values

    base_pt, base_lo, base_hi, base_auc = ci(logreg_baseline, df, y)
    print(f"baseline LogReg  PR-AUC={base_pt:.3f}  CI=[{base_lo:.3f}, {base_hi:.3f}]")

    print(f"\ntuning XGBoost ({n_trials} trials)...")
    best_params, best_value, study = tune_xgb(df, y, n_trials=n_trials)
    print(f"best CV PR-AUC during search = {best_value:.3f}")
    print(f"best params = {best_params}")

    xgb_pt, xgb_lo, xgb_hi, xgb_auc = ci(lambda: build_xgb(best_params, n_jobs=1), df, y)
    print(f"\nXGBoost tuned    PR-AUC={xgb_pt:.3f}  CI=[{xgb_lo:.3f}, {xgb_hi:.3f}]  ROC-AUC={xgb_auc:.3f}")

    non_overlap = xgb_lo > base_hi  # XGB CI strictly above baseline CI
    print(
        "\nVERDICT H2: "
        + (
            f"SUPPORTED — XGBoost CI [{xgb_lo:.3f},{xgb_hi:.3f}] does not overlap "
            f"baseline CI [{base_lo:.3f},{base_hi:.3f}]; new champion."
            if non_overlap
            else f"REJECTED — XGBoost CI [{xgb_lo:.3f},{xgb_hi:.3f}] overlaps baseline "
            f"CI [{base_lo:.3f},{base_hi:.3f}]; keep the simpler LogReg."
        )
    )

    with start_run("h2-xgb-tuned"):
        mlflow.log_params(best_params)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_metric("pr_auc", xgb_pt)
        mlflow.log_metric("pr_auc_ci_low", xgb_lo)
        mlflow.log_metric("pr_auc_ci_high", xgb_hi)
        mlflow.log_metric("roc_auc", xgb_auc)
        mlflow.log_metric("baseline_pr_auc", base_pt)
        mlflow.log_metric("baseline_pr_auc_ci_low", base_lo)
        mlflow.log_metric("baseline_pr_auc_ci_high", base_hi)
        mlflow.set_tag("h2_supported", str(non_overlap))


if __name__ == "__main__":
    main()
