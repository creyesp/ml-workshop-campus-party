from floodit import config
from floodit.data.load_v4 import load_v4


def test_v4_join_keeps_split_and_labels():
    tr = load_v4("train", verify=False)
    assert len(tr) == 7190
    assert round(tr["churned"].mean(), 3) == 0.231


def test_v4_nav_features_present_no_nulls():
    tr = load_v4("train", verify=False)
    for col in config.V4_NUMERICAL_COLUMNS:
        assert col in tr.columns, f"{col} missing"
    assert not tr[config.V4_NUMERICAL_COLUMNS].isna().any().any()


def test_users_without_screens_zero_filled():
    tr = load_v4("train", verify=False)
    # 30 train users had no screen views -> nav_screen_views == 0 for some
    assert (tr["nav_screen_views"] == 0).any()


def test_reached_flags_binary():
    tr = load_v4("train", verify=False)
    assert set(tr["nav_reached_shop"].unique()) <= {0, 1}
    assert set(tr["nav_reached_steps"].unique()) <= {0, 1}
