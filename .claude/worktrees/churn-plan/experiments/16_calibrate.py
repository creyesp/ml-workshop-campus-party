"""Calibrate the champion (v3 XGBoost) output (spec §4 — usable probabilities).

The champion uses scale_pos_weight=4.1, which inflates probabilities. We wrap it
in CalibratedClassifierCV (isotonic and sigmoid, internal 5-fold on train) and
compare Brier + ECE + reliability on the frozen test vs. the raw model. Ranking
metrics (PR-AUC/ROC-AUC) are preserved because both calibrators are monotonic.

Run: .venv/bin/python experiments/16_calibrate.py
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from floodit import config
from floodit.data.load_v3 import load_v3
from floodit.evaluate.metrics import brier, pr_auc, roc_auc
from floodit.features.preprocess import make_preprocessor

V3 = config.V3_NUMERICAL_COLUMNS
CAT = config.CATEGORICAL_COLUMNS
ART = "experiments/_artifacts"


def champion_v3():
    best = json.load(open(f"{ART}/xgb_v3_best_params.json"))
    est = XGBClassifier(tree_method="hist", eval_metric="logloss",
                        random_state=config.RANDOM_SEED, n_jobs=1, **best)
    return Pipeline([("pre", make_preprocessor(V3, CAT)), ("clf", est)])


def ece(y, p, bins=10):
    """Expected Calibration Error (equal-width bins)."""
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum():
            e += abs(p[m].mean() - y[m].mean()) * m.sum() / len(y)
    return e


def main():
    train = load_v3("train", verify=True)
    test = load_v3("test", verify=True)
    ytr = train[config.LABEL_COLUMN].values
    yte = test[config.LABEL_COLUMN].values

    models = {}
    # raw (uncalibrated)
    raw = champion_v3().fit(train, ytr)
    models["uncalibrated"] = raw.predict_proba(test)[:, 1]
    # calibrated variants
    for method in ("sigmoid", "isotonic"):
        cal = CalibratedClassifierCV(champion_v3(), method=method, cv=5,
                                     n_jobs=config.N_JOBS)
        cal.fit(train, ytr)
        models[method] = cal.predict_proba(test)[:, 1]
        if method == "isotonic":
            joblib.dump(cal, f"{ART}/champion_v3_calibrated_isotonic.joblib")

    print(f"{'variant':14s} {'Brier':>7s} {'ECE':>7s} {'PR-AUC':>7s} {'ROC-AUC':>8s}")
    rows = {}
    for name, p in models.items():
        rows[name] = dict(brier=brier(yte, p), ece=ece(yte, p),
                          pr_auc=pr_auc(yte, p), roc_auc=roc_auc(yte, p))
        r = rows[name]
        print(f"{name:14s} {r['brier']:7.4f} {r['ece']:7.4f} {r['pr_auc']:7.3f} {r['roc_auc']:8.3f}")
    json.dump(rows, open(f"{ART}/calibration_metrics.json", "w"), indent=2)

    best_cal = min(("sigmoid", "isotonic"), key=lambda m: rows[m]["brier"])
    print(f"\nbest calibrator by Brier: {best_cal} "
          f"(Brier {rows['uncalibrated']['brier']:.4f} -> {rows[best_cal]['brier']:.4f}, "
          f"ECE {rows['uncalibrated']['ece']:.4f} -> {rows[best_cal]['ece']:.4f})")
    # cv=5 CalibratedClassifierCV averages 5 fold-models (an ensemble), so
    # ranking is not just preserved by the monotonic map — it can improve.
    d_pr = rows[best_cal]["pr_auc"] - rows["uncalibrated"]["pr_auc"]
    d_roc = rows[best_cal]["roc_auc"] - rows["uncalibrated"]["roc_auc"]
    print(f"ranking (cv-ensemble): PR-AUC {d_pr:+.3f}, ROC-AUC {d_roc:+.3f} vs uncalibrated")

    # reliability diagram
    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], "k:", label="perfectly calibrated")
    for name in ("uncalibrated", "sigmoid", "isotonic"):
        frac_pos, mean_pred = calibration_curve(yte, models[name], n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o",
                 label=f"{name} (Brier {rows[name]['brier']:.3f})")
    plt.xlabel("mean predicted probability")
    plt.ylabel("observed churn fraction")
    plt.title("v3 champion — reliability diagram (frozen test)")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.savefig(f"{ART}/calibration_reliability_v3.png", dpi=110, bbox_inches="tight")
    print(f"saved {ART}/calibration_reliability_v3.png")


if __name__ == "__main__":
    main()
