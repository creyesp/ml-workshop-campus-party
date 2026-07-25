import numpy as np

from floodit import config
from floodit.data.load import load_dataset
from floodit.models.split import stratified_kfold, time_based_split


def test_stratified_kfold_preserves_prevalence_and_is_disjoint():
    df = load_dataset("train", verify=False)
    y = df[config.LABEL_COLUMN].values
    folds = list(stratified_kfold(df, y, n_splits=5))
    assert len(folds) == 5
    seen_val = []
    for tr, va in folds:
        assert abs(y[va].mean() - y.mean()) < 0.03
        assert set(tr).isdisjoint(set(va))
        seen_val.append(va)
    # every index appears in exactly one validation fold
    all_val = np.concatenate(seen_val)
    assert sorted(all_val.tolist()) == list(range(len(df)))


def test_stratified_kfold_is_deterministic():
    df = load_dataset("train", verify=False)
    y = df[config.LABEL_COLUMN].values
    a = [va.tolist() for _, va in stratified_kfold(df, y, n_splits=5)]
    b = [va.tolist() for _, va in stratified_kfold(df, y, n_splits=5)]
    assert a == b


def test_time_split_is_ordered_by_first_engagement():
    df = load_dataset("train", verify=False)
    tr, va = time_based_split(df, frac=0.8)
    t = df["user_first_engagement"]
    assert t.iloc[tr].max() <= t.iloc[va].min()
    assert len(tr) + len(va) == len(df)
    assert abs(len(tr) / len(df) - 0.8) < 0.01
