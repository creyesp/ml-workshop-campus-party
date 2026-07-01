"""Confirm the v3 (session-level) champion on the frozen test + cost point.

Refits tuned-XGBoost-on-v3 on full train, scores users_test.csv once, compares
to the v2 champion (test PR-AUC 0.460), and reports the cost-optimal operating
point (FN = 3-4x FP).

Run: .venv/bin/python experiments/11_final_v3.py
"""
import json

import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from floodit import config
from floodit.data.load_v3 import load_v3
from floodit.evaluate.metrics import bootstrap_ci, brier, pr_auc, roc_auc
from floodit.features.preprocess import make_preprocessor
from floodit.models.scoring import oof_proba

V3 = config.V3_NUMERICAL_COLUMNS
CAT = config.CATEGORICAL_COLUMNS


def champion_v3():
    best = json.load(open("experiments/_artifacts/xgb_v3_best_params.json"))
    est = XGBClassifier(tree_method="hist", eval_metric="logloss",
                        random_state=config.RANDOM_SEED, n_jobs=1, **best)
    return Pipeline([("pre", make_preprocessor(V3, CAT)), ("clf", est)])


def best_cost_threshold(y, prob, c):
    grid = np.unique(np.concatenate([[0.0], np.sort(prob), [1.0]]))
    costs = [(((prob >= t) & (y == 0)).sum() + c * ((prob < t) & (y == 1)).sum()) for t in grid]
    return float(grid[int(np.argmin(costs))])


def main():
    train = load_v3("train", verify=True)
    test = load_v3("test", verify=True)
    ytr = train[config.LABEL_COLUMN].values
    yte = test[config.LABEL_COLUMN].values

    model = champion_v3().fit(train, ytr)
    prob = model.predict_proba(test)[:, 1]
    pt = pr_auc(yte, prob)
    lo, mid, hi = bootstrap_ci(yte, prob, metric=pr_auc, n=1000)
    print(f"v3 champion TEST  PR-AUC={pt:.3f}  CI=[{lo:.3f}, {hi:.3f}]  "
          f"ROC-AUC={roc_auc(yte, prob):.3f}  Brier={brier(yte, prob):.3f}")
    print("v2 champion TEST  PR-AUC=0.460  CI=[0.394, 0.540]  ROC-AUC=0.768")

    _, oof = oof_proba(champion_v3, train, ytr, n_splits=5)
    for c in (3.0, 4.0):
        t = best_cost_threshold(ytr, oof, c)
        pred = (prob >= t).astype(int)
        fp = int(((pred == 1) & (yte == 0)).sum())
        fn = int(((pred == 0) & (yte == 1)).sum())
        cost = fp + c * fn
        best_naive = min(c * int((yte == 1).sum()), int((yte == 0).sum()))
        print(f"  c={c:.0f}: thr={t:.3f} precision={precision_score(yte, pred, zero_division=0):.3f} "
              f"recall={recall_score(yte, pred, zero_division=0):.3f} cost={cost:.0f} "
              f"(saves {best_naive - cost:.0f}, {100*(best_naive-cost)/best_naive:.0f}% vs best naive)")


if __name__ == "__main__":
    main()
