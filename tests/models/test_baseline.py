import numpy as np

from floodit.data.load import load_dataset
from floodit.models.baseline import logreg_baseline, majority_baseline


def test_majority_predicts_negative_class():
    df = load_dataset("train", verify=False)
    y = df["churned"].values
    clf = majority_baseline().fit(df, y)
    preds = clf.predict(df)
    assert set(np.unique(preds)) == {0}  # majority class is "retained" (0)
    # predict_proba for positive class ~ 0 (never predicts churn)
    proba = clf.predict_proba(df)[:, 1]
    assert proba.max() <= 0.5


def test_logreg_is_class_balanced_and_seeded():
    pipe = logreg_baseline()
    clf = pipe.named_steps["clf"]
    assert clf.class_weight == "balanced"
    assert clf.random_state == 42


def test_logreg_fits_and_scores_above_prevalence():
    df = load_dataset("train", verify=False)
    y = df["churned"].values
    pipe = logreg_baseline().fit(df, y)
    proba = pipe.predict_proba(df)[:, 1]
    # in-sample sanity: AUC clearly above chance
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y, proba) > 0.6
