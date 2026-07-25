"""Learning curve for the v3 champion: does training-set size still help?

For a grid of training-set fractions, draw R stratified subsamples (seeded per
repeat), fit the v3 champion pipeline, and score PR-AUC on (a) the frozen test
set (generalization curve) and (b) the training subsample (overfitting gap).
Aggregates mean/std across repeats and saves a plot.

This informs the DNN question: a plateaued curve at ~7k rows means a
data-hungry sequence DNN is unlikely to help; a still-rising curve means more
data is the lever.

Capped parallelism: estimator n_jobs=1, orchestration sequential.

Run: .venv/bin/python experiments/13_learning_curve.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from floodit import config
from floodit.data.load_v3 import load_v3
from floodit.evaluate.metrics import pr_auc
from floodit.features.preprocess import make_preprocessor

V3 = config.V3_NUMERICAL_COLUMNS
CAT = config.CATEGORICAL_COLUMNS
SEED = config.RANDOM_SEED

FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0]
R = 5  # repeats per fraction
BASE_SEED = 1000

V3_CHAMP_TEST_PR_AUC = 0.591  # task 11 frozen-test champion
PREVALENCE_FLOOR = 0.231  # positive rate (random-classifier PR-AUC floor)


def champion_v3():
    best = json.load(open("experiments/_artifacts/xgb_v3_best_params.json"))
    est = XGBClassifier(tree_method="hist", eval_metric="logloss",
                        random_state=SEED, n_jobs=1, **best)
    return Pipeline([("pre", make_preprocessor(V3, CAT)), ("clf", est)])


def stratified_subsample(df, y, frac, seed):
    """Stratified subsample of ``frac`` of the rows; frac==1.0 returns all."""
    if frac >= 1.0:
        return df, y
    # train_test_split with stratify is a seeded stratified draw.
    sub_df, _, sub_y, _ = train_test_split(
        df, y, train_size=frac, stratify=y,
        random_state=np.random.default_rng(seed).integers(0, 2**31 - 1),
    )
    return sub_df, sub_y


def main():
    train = load_v3("train", verify=True)
    test = load_v3("test", verify=True)
    ytr = train[config.LABEL_COLUMN].values
    yte = test[config.LABEL_COLUMN].values

    rows = []
    print(f"Learning curve: {len(FRACTIONS)} fractions x {R} repeats = "
          f"{len(FRACTIONS) * R} fits\n")
    for frac in FRACTIONS:
        test_scores, train_scores, n_used = [], [], None
        for r in range(R):
            seed = BASE_SEED + r
            sub_df, sub_y = stratified_subsample(train, ytr, frac, seed)
            n_used = len(sub_df)
            model = champion_v3().fit(sub_df, sub_y)
            test_scores.append(pr_auc(yte, model.predict_proba(test)[:, 1]))
            train_scores.append(pr_auc(sub_y, model.predict_proba(sub_df)[:, 1]))
        rows.append(dict(
            n_train=n_used, frac=frac,
            test_mean=float(np.mean(test_scores)),
            test_std=float(np.std(test_scores)),
            test_lo=float(np.percentile(test_scores, 2.5)),
            test_hi=float(np.percentile(test_scores, 97.5)),
            train_mean=float(np.mean(train_scores)),
            train_std=float(np.std(train_scores)),
        ))
        print(f"  frac={frac:.2f} n={n_used:5d}  "
              f"test={rows[-1]['test_mean']:.3f}+-{rows[-1]['test_std']:.3f}  "
              f"train={rows[-1]['train_mean']:.3f}")

    # --- table ---
    print("\n=== Learning curve (R={} stratified repeats per fraction) ===".format(R))
    print(f"{'n_train':>8s} {'frac':>5s} {'test_pr_auc_mean':>17s} "
          f"{'test_pr_auc_std':>16s} {'train_pr_auc_mean':>18s}")
    for row in rows:
        print(f"{row['n_train']:8d} {row['frac']:5.2f} "
              f"{row['test_mean']:17.3f} {row['test_std']:16.3f} "
              f"{row['train_mean']:18.3f}")

    # --- slope at high-data end (80% -> 100%) ---
    r80 = next(r for r in rows if abs(r["frac"] - 0.8) < 1e-9)
    r100 = next(r for r in rows if abs(r["frac"] - 1.0) < 1e-9)
    gain_80_100 = r100["test_mean"] - r80["test_mean"]
    gap_100 = r100["train_mean"] - r100["test_mean"]
    print(f"\nTest PR-AUC gain 80% -> 100% : {gain_80_100:+.4f}")
    print(f"Train-vs-test gap at 100%    : {gap_100:+.4f}")

    # --- plot ---
    fracs = [r["frac"] for r in rows]
    ns = [r["n_train"] for r in rows]
    test_m = np.array([r["test_mean"] for r in rows])
    test_s = np.array([r["test_std"] for r in rows])
    train_m = np.array([r["train_mean"] for r in rows])
    train_s = np.array([r["train_std"] for r in rows])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(ns, train_m, "o-", color="#d1495b", label="Train PR-AUC")
    ax.fill_between(ns, train_m - train_s, train_m + train_s, color="#d1495b", alpha=0.15)
    ax.plot(ns, test_m, "o-", color="#1b6ca8", label="Test PR-AUC (frozen)")
    ax.fill_between(ns, test_m - test_s, test_m + test_s, color="#1b6ca8", alpha=0.15)
    ax.axhline(V3_CHAMP_TEST_PR_AUC, ls="--", color="#2a9d8f",
               label=f"v3 champion test ({V3_CHAMP_TEST_PR_AUC})")
    ax.axhline(PREVALENCE_FLOOR, ls=":", color="grey",
               label=f"prevalence floor ({PREVALENCE_FLOOR})")
    ax.set_xlabel("Training-set size (rows)")
    ax.set_ylabel("PR-AUC")
    ax.set_title("v3 champion learning curve (PR-AUC vs. train size)")
    ax.legend(loc="center right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = "experiments/_artifacts/learning_curve_v3.png"
    fig.savefig(out, dpi=130)
    print(f"\nSaved plot -> {out}")

    json.dump(rows, open("experiments/_artifacts/learning_curve_v3.json", "w"), indent=2)


if __name__ == "__main__":
    main()
