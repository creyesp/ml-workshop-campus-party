from floodit import config
from floodit.data.load_v2 import load_v2


def test_v2_join_keeps_split_rows_and_labels():
    tr = load_v2("train", verify=False)
    te = load_v2("test", verify=False)
    assert len(tr) == 7190
    assert len(te) == 799
    assert round(tr["churned"].mean(), 3) == 0.231


def test_v2_numeric_features_present_and_no_nulls():
    tr = load_v2("train", verify=False)
    for col in config.V2_NUMERICAL_COLUMNS:
        assert col in tr.columns, f"{col} missing"
    assert not tr[config.V2_NUMERICAL_COLUMNS].isna().any().any()


def test_num_sessions_dropped():
    tr = load_v2("train", verify=False)
    assert "num_sessions" not in tr.columns


def test_never_times_imputed_to_sentinel():
    tr = load_v2("train", verify=False)
    # the sentinel must appear (some users never complete a level in-window)
    assert (tr["min_to_first_level_complete"] == config.NEVER_SENTINEL).any()
