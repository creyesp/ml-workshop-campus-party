"""Baselines (spec §5, datapowers:baseline-first-modeling).

- ``majority_baseline``: trivial constant predictor (always "retained"). Its
  PR-AUC floor is the prevalence (~0.23); any real model must beat it.
- ``logreg_baseline``: the *strong* baseline — class-balanced Logistic
  Regression on the standard preprocessor. Re-scored under the new protocol
  (PR-AUC + bootstrap CI + CV), this becomes the documented number to beat.
"""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from floodit import config
from floodit.features.preprocess import build_preprocessor


def majority_baseline() -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent")


def logreg_baseline() -> Pipeline:
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
