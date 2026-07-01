"""Baselines under the new protocol (spec §5): CV PR-AUC + bootstrap CI.

Scores the trivial (majority) and strong (Logistic Regression) baselines with
5-fold out-of-fold predictions on the training set, bootstraps a PR-AUC CI, and
logs both to MLflow. The strong-baseline CI is the documented number to beat.

Run: .venv/bin/python experiments/02_baselines.py
"""
import mlflow

from floodit import config
from floodit.data.contract import validate_contract
from floodit.data.load import load_dataset
from floodit.evaluate.metrics import bootstrap_ci, pr_auc, roc_auc
from floodit.models.baseline import logreg_baseline, majority_baseline
from floodit.models.scoring import oof_proba
from floodit.tracking import start_run


def score(name, factory, df, y):
    yy, oof = oof_proba(factory, df, y, n_splits=5)
    lo, mid, hi = bootstrap_ci(yy, oof, metric=pr_auc, n=1000)
    point = pr_auc(yy, oof)
    auc = roc_auc(yy, oof)
    with start_run(f"baseline-{name}"):
        mlflow.log_param("model", name)
        mlflow.log_param("cv_splits", 5)
        mlflow.log_metric("pr_auc", point)
        mlflow.log_metric("pr_auc_ci_low", lo)
        mlflow.log_metric("pr_auc_ci_high", hi)
        mlflow.log_metric("roc_auc", auc)
    print(
        f"{name:10s}  PR-AUC={point:.3f}  CI=[{lo:.3f}, {hi:.3f}]  ROC-AUC={auc:.3f}"
    )
    return point, lo, hi


def main():
    df = load_dataset("train", verify=True)
    validate_contract(df)
    y = df[config.LABEL_COLUMN].values
    print(f"prevalence (PR-AUC floor) = {y.mean():.3f}\n")
    score("majority", majority_baseline, df, y)
    score("logreg", logreg_baseline, df, y)


if __name__ == "__main__":
    main()
