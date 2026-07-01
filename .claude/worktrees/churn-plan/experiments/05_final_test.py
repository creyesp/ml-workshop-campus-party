"""Final held-out scoring on the frozen test set (spec §6, §4, §8).

Scores the strong baseline and the XGBoost champion ONCE on users_test.csv.
The operating threshold is chosen on TRAIN out-of-fold predictions (never on
test) so the test estimate stays honest. Reports PR-AUC + bootstrap CI,
ROC-AUC, Brier, the operating point, a confusion matrix, segment metrics, and a
time-split robustness check for non-stationarity (Q3).

Run: .venv/bin/python experiments/05_final_test.py
"""
import json

import mlflow
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from floodit import config
from floodit.data.contract import validate_contract
from floodit.data.load import load_dataset
from floodit.evaluate.metrics import (
    bootstrap_ci,
    brier,
    pr_auc,
    recall_at_precision,
    roc_auc,
    threshold_at_precision,
)
from floodit.evaluate.segments import segment_metrics
from floodit.models.baseline import logreg_baseline
from floodit.models.scoring import oof_proba
from floodit.models.split import time_based_split
from floodit.models.xgb import build_xgb
from floodit.tracking import start_run

PARAMS_FILE = "experiments/_artifacts/xgb_best_params.json"
TARGET_PRECISION = 0.60


def champion_factory():
    params = json.load(open(PARAMS_FILE))
    return build_xgb(params)


def score_on_test(name, model, train_df, ytr, test_df, yte):
    model.fit(train_df, ytr)
    prob = model.predict_proba(test_df)[:, 1]
    pt = pr_auc(yte, prob)
    lo, mid, hi = bootstrap_ci(yte, prob, metric=pr_auc, n=1000)
    print(
        f"{name:10s}  test PR-AUC={pt:.3f}  CI=[{lo:.3f}, {hi:.3f}]  "
        f"ROC-AUC={roc_auc(yte, prob):.3f}  Brier={brier(yte, prob):.3f}"
    )
    return prob, (pt, lo, hi)


def main():
    train = load_dataset("train", verify=True)
    test = load_dataset("test", verify=True)
    validate_contract(train)
    validate_contract(test)
    ytr = train[config.LABEL_COLUMN].values
    yte = test[config.LABEL_COLUMN].values
    print(f"train n={len(train)} ({ytr.mean():.3f} churn) | test n={len(test)} ({yte.mean():.3f} churn)\n")

    base_prob, base_ci = score_on_test("logreg", logreg_baseline(), train, ytr, test, yte)
    champ_prob, champ_ci = score_on_test("xgboost", champion_factory(), train, ytr, test, yte)

    # --- Operating point chosen on TRAIN OOF (not test) -------------------
    _, champ_oof = oof_proba(champion_factory, train, ytr, n_splits=5)
    thr = threshold_at_precision(ytr, champ_oof, target=TARGET_PRECISION)
    train_r_at_p = recall_at_precision(ytr, champ_oof, target=TARGET_PRECISION)
    print(f"\noperating threshold (train OOF, precision>={TARGET_PRECISION}): {thr:.3f} "
          f"(train recall@p = {train_r_at_p:.3f})")

    pred = (champ_prob >= thr).astype(int)
    p_test = precision_score(yte, pred, zero_division=0)
    r_test = recall_score(yte, pred, zero_division=0)
    cm = confusion_matrix(yte, pred)
    print(f"champion @thr on TEST: precision={p_test:.3f} recall={r_test:.3f}")
    print(f"confusion matrix [tn fp; fn tp]:\n{cm}")
    test_r_at_p = recall_at_precision(yte, champ_prob, target=TARGET_PRECISION)
    print(f"champion test recall@precision>={TARGET_PRECISION}: {test_r_at_p:.3f}")

    # --- Robustness: time-based split (non-stationarity, Q3) --------------
    tr_idx, va_idx = time_based_split(train, frac=0.8)
    m = champion_factory()
    m.fit(train.iloc[tr_idx], ytr[tr_idx])
    time_prob = m.predict_proba(train.iloc[va_idx])[:, 1]
    time_prauc = pr_auc(ytr[va_idx], time_prob)
    drop = champ_ci[0] - time_prauc
    flag = "DROP => possible non-stationarity" if drop > 0.05 else "no degradation"
    print(f"\nrobustness — time-split (early->late) champion PR-AUC = {time_prauc:.3f} "
          f"(vs random-CV test {champ_ci[0]:.3f}); {flag}")

    # --- Segment metrics on test (fairness visibility, spec §8) -----------
    print("\nsegment PR-AUC (champion, test) by device_os:")
    print(segment_metrics(test, yte, champ_prob, by="device_os").to_string(index=False))
    print("\ntop country segments (champion, test):")
    print(segment_metrics(test, yte, champ_prob, by="country_name").head(8).to_string(index=False))

    with start_run("final-test-champion"):
        mlflow.set_tag("final_eval", "true")
        mlflow.log_metric("test_pr_auc", champ_ci[0])
        mlflow.log_metric("test_pr_auc_ci_low", champ_ci[1])
        mlflow.log_metric("test_pr_auc_ci_high", champ_ci[2])
        mlflow.log_metric("baseline_test_pr_auc", base_ci[0])
        mlflow.log_metric("operating_threshold", thr)
        mlflow.log_metric("test_precision_at_thr", p_test)
        mlflow.log_metric("test_recall_at_thr", r_test)
        mlflow.log_metric("test_recall_at_precision_target", test_r_at_p)
        mlflow.log_metric("time_split_pr_auc", time_prauc)


if __name__ == "__main__":
    main()
