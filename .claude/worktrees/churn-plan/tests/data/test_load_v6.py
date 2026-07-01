from floodit import config
from floodit.data.load_v6 import load_v6


def test_v6_join_keeps_split_and_labels():
    tr = load_v6("train", verify=False)
    assert len(tr) == 7190
    assert round(tr["churned"].mean(), 3) == 0.231


def test_v6_difficulty_features_present_no_nulls():
    tr = load_v6("train", verify=False)
    for col in config.V6_NUMERICAL_COLUMNS:
        assert col in tr.columns, f"{col} missing"
    assert not tr[config.V6_NUMERICAL_COLUMNS].isna().any().any()


def test_fail_signal_has_variance():
    tr = load_v6("train", verify=False)
    assert tr["diff_cnt_fail"].max() > 0
    assert (tr["diff_cnt_fail"] == 0).any()


def test_max_level_nulls_imputed_zero():
    tr = load_v6("train", verify=False)
    assert tr["diff_max_level"].min() == 0
