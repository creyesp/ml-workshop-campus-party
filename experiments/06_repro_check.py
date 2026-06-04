"""Reproducibility verification (spec §8, datapowers:reproducibility-verification).

Re-runs the baseline and champion CV scoring end-to-end and asserts the metrics
match the values recorded during the experiment, within tolerance. Same code +
hash-verified data + fixed seeds must reproduce the same numbers.

Run: .venv/bin/python experiments/06_repro_check.py
"""
import json

from floodit import config
from floodit.data.contract import validate_contract
from floodit.data.load import load_dataset
from floodit.evaluate.metrics import pr_auc
from floodit.models.baseline import logreg_baseline
from floodit.models.scoring import oof_proba
from floodit.models.xgb import build_xgb

TOL = 0.005
EXPECTED = {"logreg": 0.315, "xgboost": 0.400}  # logged CV PR-AUC (tasks 10, 12)


def champion_factory():
    params = json.load(open("experiments/_artifacts/xgb_best_params.json"))
    return build_xgb(params, n_jobs=1)


def main():
    df = load_dataset("train", verify=True)  # hash-pinned
    validate_contract(df)
    y = df[config.LABEL_COLUMN].values

    results = {}
    for name, factory in (("logreg", logreg_baseline), ("xgboost", champion_factory)):
        yy, oof = oof_proba(factory, df, y, n_splits=5)
        results[name] = pr_auc(yy, oof)

    ok = True
    for name, expected in EXPECTED.items():
        got = results[name]
        delta = abs(got - expected)
        status = "MATCH" if delta <= TOL else "MISMATCH"
        ok = ok and delta <= TOL
        print(f"{name:10s} expected={expected:.3f} got={got:.3f} |Δ|={delta:.4f} -> {status} (tol {TOL})")

    print("\nREPRODUCIBILITY:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
