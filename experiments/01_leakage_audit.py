"""H3 leakage audit (spec §7/§8) — the experiment's tripwire.

Runs univariate AUC + ablation for every cnt_* feature against the churn label
on the training set, prints the table, and applies the stop rule:

  TRIPWIRE if any feature has directed ROC-AUC >= 0.85, OR neutralising a single
  feature collapses CV PR-AUC toward prevalence (ablated <= prevalence + 0.02
  while full is materially higher).

Run: .venv/bin/python experiments/01_leakage_audit.py
"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from floodit import config
from floodit.data.contract import validate_contract
from floodit.data.load import load_dataset
from floodit.evaluate.leakage import audit_all
from floodit.features.preprocess import build_preprocessor

AUC_TRIPWIRE = 0.85


def logreg_factory():
    return Pipeline(
        steps=[
            ("pre", build_preprocessor()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=config.RANDOM_SEED,
                ),
            ),
        ]
    )


def main() -> None:
    df = load_dataset("train", verify=True)
    validate_contract(df)
    y = df[config.LABEL_COLUMN].values
    prevalence = float(y.mean())

    table = audit_all(logreg_factory, df, y)
    with __import__("pandas").option_context("display.width", 160, "display.max_columns", None):
        print(table.to_string(index=False))

    flagged_auc = table[table["directed_auc"] >= AUC_TRIPWIRE]["feature"].tolist()
    flagged_abl = table[
        (table["cv_pr_auc_ablated"] <= prevalence + 0.02)
        & (table["ablation_delta"] > 0.10)
    ]["feature"].tolist()

    print(f"\nprevalence = {prevalence:.3f}")
    print(f"directed_auc >= {AUC_TRIPWIRE}: {flagged_auc or 'none'}")
    print(f"ablation-collapse features: {flagged_abl or 'none'}")
    if flagged_auc or flagged_abl:
        print("\nVERDICT: TRIPWIRE — halt modeling, redefine flagged features (spec §10 Kill).")
    else:
        print("\nVERDICT: no disqualifying leakage — proceed to baselines.")


if __name__ == "__main__":
    main()
