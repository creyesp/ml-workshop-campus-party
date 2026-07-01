"""H1 — cost/precision-based operating point vs. the default 0.5 (spec §7 H1, §4).

Falsifiable check: selecting the decision threshold by a precision target
(>= 0.60, spec §4 placeholder) improves churn **recall @ precision >= 0.60**
over the default 0.5-threshold operating point. If it does not, reject H1.

Uses out-of-fold predictions from the strong (LogReg) baseline so the threshold
is chosen on held-out folds, not in-sample.

Run: .venv/bin/python experiments/03_threshold_tuning.py
"""
import mlflow
import numpy as np
from sklearn.metrics import precision_score, recall_score

from floodit import config
from floodit.data.contract import validate_contract
from floodit.data.load import load_dataset
from floodit.evaluate.metrics import (
    bootstrap_ci,
    recall_at_precision,
    threshold_at_precision,
)
from floodit.models.baseline import logreg_baseline
from floodit.models.scoring import oof_proba
from floodit.tracking import start_run

TARGET_PRECISION = 0.60


def _recall_at_p(target):
    def metric(y, p):
        return recall_at_precision(y, p, target=target)

    return metric


def main():
    df = load_dataset("train", verify=True)
    validate_contract(df)
    y = df[config.LABEL_COLUMN].values
    yy, oof = oof_proba(logreg_baseline, df, y, n_splits=5)

    # Default operating point: 0.5
    pred05 = (oof >= 0.5).astype(int)
    p05 = precision_score(yy, pred05, zero_division=0)
    r05 = recall_score(yy, pred05, zero_division=0)

    # Cost-tuned operating point: smallest threshold with precision >= target
    thr = threshold_at_precision(yy, oof, target=TARGET_PRECISION)
    predt = (oof >= thr).astype(int)
    pt = precision_score(yy, predt, zero_division=0)
    rt = recall_score(yy, predt, zero_division=0)
    r_at_p = recall_at_precision(yy, oof, target=TARGET_PRECISION)

    lo, mid, hi = bootstrap_ci(yy, oof, metric=_recall_at_p(TARGET_PRECISION), n=1000)

    print(f"target precision = {TARGET_PRECISION}\n")
    print(f"{'rule':16s} {'threshold':>9s} {'precision':>10s} {'recall':>8s}")
    print(f"{'default 0.5':16s} {0.50:9.2f} {p05:10.3f} {r05:8.3f}")
    print(f"{'cost-tuned':16s} {thr:9.2f} {pt:10.3f} {rt:8.3f}")
    print(f"\nrecall @ precision>={TARGET_PRECISION}: {r_at_p:.3f}  CI=[{lo:.3f}, {hi:.3f}]")

    meets_05 = p05 >= TARGET_PRECISION
    verdict_pass = (rt > r05) or (not meets_05 and pt >= TARGET_PRECISION and rt > 0)
    print(
        "\nVERDICT H1: "
        + (
            "SUPPORTED — tuned threshold gives a usable precision>=0.60 operating "
            f"point (recall {rt:.3f}) that the default 0.5 does not."
            if verdict_pass
            else "REJECTED — tuning does not improve recall@precision>=0.60 over 0.5."
        )
    )

    with start_run("h1-threshold-tuning"):
        mlflow.log_param("target_precision", TARGET_PRECISION)
        mlflow.log_metric("precision_at_0.5", p05)
        mlflow.log_metric("recall_at_0.5", r05)
        mlflow.log_metric("tuned_threshold", thr)
        mlflow.log_metric("precision_tuned", pt)
        mlflow.log_metric("recall_tuned", rt)
        mlflow.log_metric("recall_at_precision_target", r_at_p)
        mlflow.log_metric("recall_at_p_ci_low", lo)
        mlflow.log_metric("recall_at_p_ci_high", hi)
        mlflow.set_tag("h1_supported", str(verdict_pass))


if __name__ == "__main__":
    main()
