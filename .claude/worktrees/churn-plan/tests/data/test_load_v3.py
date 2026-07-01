from floodit import config
from floodit.data.load_v3 import load_v3


def test_v3_join_keeps_split_and_labels():
    tr = load_v3("train", verify=False)
    assert len(tr) == 7190
    assert round(tr["churned"].mean(), 3) == 0.231


def test_v3_session_features_present_no_nulls():
    tr = load_v3("train", verify=False)
    for col in config.V3_NUMERICAL_COLUMNS:
        assert col in tr.columns, f"{col} missing"
    assert not tr[config.V3_NUMERICAL_COLUMNS].isna().any().any()


def test_sess_returned_is_binary():
    tr = load_v3("train", verify=False)
    assert set(tr["sess_returned"].unique()) <= {0, 1}


def test_never_second_session_imputed():
    tr = load_v3("train", verify=False)
    assert (tr["sess_min_to_second_session"] == config.NEVER_SENTINEL).any()
