"""Confirm the v2 champion's lift on the frozen test set + cost operating point.

Refits tuned-XGBoost-on-v2 on full train, scores users_test.csv once, and
compares to the v1 champion's test PR-AUC (0.384). Also reports the cost-optimal
operating point (FN = 3-4x FP) for the v2 champion.

Run: .venv/bin/python experiments/09_final_v2.py
"""
import json

import numpy as np
from sklearn.metrics import precision_score, recall_score
from xgboost import XGBClassifier

from floodit import config
from floodit.data.load_v2 import load_v2
from floodit.evaluate.metrics import bootstrap_ci, brier, pr_auc, roc_auc
from floodit.features.preprocess import make_preprocessor
from floodit.models.scoring import oof_proba
from sklearn.pipeline import Pipeline

V2 = config.V2_NUMERICAL_COLUMNS
CAT = config.CATEGORICAL_COLUMNS


def champion_v2():
    best = json.load(open("experiments/_artifacts/xgb_v2_best_params.json"))
    est = XGBClassifier(tree_method="hist", eval_metric="logloss",
                        random_state=config.RANDOM_SEED, n_jobs=1, **best)
    return Pipeline([("pre", make_preprocessor(V2, CAT)), ("clf", est)])


def best_cost_threshold(y, prob, c):
    grid = np.unique(np.concatenate([[0.0], np.sort(prob), [1.0]]))
    costs = [((( (prob >= t) & (y == 0)).sum()) + c * (((prob < t) & (y == 1)).sum())) for t in grid]
    return float(grid[int(np.argmin(costs))])


def main():
    train = load_v2("train", verify=True)
    test = load_v2("test", verify=True)
    ytr = train[config.LABEL_COLUMN].values
    yte = test[config.LABEL_COLUMN].values

    model = champion_v2().fit(train, ytr)
    prob = model.predict_proba(test)[:, 1]
    pt = pr_auc(yte, prob)
    lo, mid, hi = bootstrap_ci(yte, prob, metric=pr_auc, n=1000)
    print(f"v2 champion TEST  PR-AUC={pt:.3f}  CI=[{lo:.3f}, {hi:.3f}]  "
          f"ROC-AUC={roc_auc(yte, prob):.3f}  Brier={brier(yte, prob):.3f}")
    print("v1 champion TEST  PR-AUC=0.384  CI=[0.326, 0.459]  ROC-AUC=0.694")

    # Cost-optimal operating point (threshold chosen on train OOF)
    _, oof = oof_proba(champion_v2, train, ytr, n_splits=5)
    for c in (3.0, 4.0):
        t = best_cost_threshold(ytr, oof, c)
        pred = (prob >= t).astype(int)
        fp = int(((pred == 1) & (yte == 0)).sum())
        fn = int(((pred == 0) & (yte == 1)).sum())
        cost = fp + c * fn
        cost_none = c * int((yte == 1).sum())
        cost_all = int((yte == 0).sum())
        save = min(cost_none, cost_all) - cost
        print(f"  c={c:.0f}: thr={t:.3f} precision={precision_score(yte, pred, zero_division=0):.3f} "
              f"recall={recall_score(yte, pred, zero_division=0):.3f} cost={cost:.0f} "
              f"(saves {save:.0f}, {100*save/min(cost_none,cost_all):.0f}% vs best naive)")


if __name__ == "__main__":
    main()
