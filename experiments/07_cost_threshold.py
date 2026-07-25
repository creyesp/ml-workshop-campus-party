"""Cost-optimal operating point (resolves spec Q1).

Stakeholder cost ratio: a false negative (let a churner pass) costs ~3-4x a
false positive (treat a user who would have stayed). The cost-optimal threshold
is therefore well below 0.5. We pick it on TRAIN out-of-fold predictions by
minimising  cost = #FP * 1 + #FN * c , then report the operating point and
expected cost on the frozen TEST set, vs. the do-nothing and treat-everyone
baselines.

Run: .venv/bin/python experiments/07_cost_threshold.py
"""
import json

import numpy as np
from sklearn.metrics import precision_score, recall_score

from floodit import config
from floodit.data.load import load_dataset
from floodit.models.scoring import oof_proba
from floodit.models.xgb import build_xgb

COSTS = (3.0, 4.0)  # C_FN / C_FP candidates


def champion_factory():
    params = json.load(open("experiments/_artifacts/xgb_best_params.json"))
    return build_xgb(params, n_jobs=1)


def total_cost(y, pred, c):
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return fp * 1.0 + fn * c, fp, fn


def best_threshold(y, prob, c):
    grid = np.unique(np.concatenate([[0.0], np.sort(prob), [1.0]]))
    costs = [total_cost(y, (prob >= t).astype(int), c)[0] for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


def report_point(name, y, prob, t, c):
    pred = (prob >= t).astype(int)
    cost, fp, fn = total_cost(y, pred, c)
    prec = precision_score(y, pred, zero_division=0)
    rec = recall_score(y, pred, zero_division=0)
    treated = int(pred.sum())
    print(
        f"  {name:14s} thr={t:.3f}  treated={treated:4d}  "
        f"precision={prec:.3f} recall={rec:.3f}  FP={fp:4d} FN={fn:4d}  cost={cost:7.1f}"
    )
    return cost


def main():
    train = load_dataset("train", verify=True)
    test = load_dataset("test", verify=True)
    ytr = train[config.LABEL_COLUMN].values
    yte = test[config.LABEL_COLUMN].values

    _, oof = oof_proba(champion_factory, train, ytr, n_splits=5)
    model = champion_factory().fit(train, ytr)
    test_prob = model.predict_proba(test)[:, 1]

    for c in COSTS:
        t_star, _ = best_threshold(ytr, oof, c)
        formula_t = 1.0 / (1.0 + c)
        print(f"\n=== C_FN/C_FP = {c:.0f}  (cost-optimal threshold ~ 1/(1+c) = {formula_t:.2f}) ===")
        print(f"chosen threshold on train OOF: {t_star:.3f}")
        print("TEST set operating points:")
        cost_none = report_point("do-nothing", yte, test_prob, 1.01, c)   # treat none
        cost_all = report_point("treat-all", yte, test_prob, 0.0, c)      # treat everyone
        report_point("default 0.5", yte, test_prob, 0.5, c)
        cost_opt = report_point("cost-optimal", yte, test_prob, t_star, c)
        best_naive = min(cost_none, cost_all)
        print(
            f"  -> cost-optimal saves {best_naive - cost_opt:.1f} "
            f"({100*(best_naive - cost_opt)/best_naive:.0f}%) vs best naive policy "
            f"({'do-nothing' if cost_none < cost_all else 'treat-all'})"
        )


if __name__ == "__main__":
    main()
