import pytest

from floodit.data.contract import validate_contract
from floodit.data.load import load_dataset


def test_real_train_passes_contract():
    validate_contract(load_dataset("train", verify=False))  # must not raise


def test_real_test_passes_contract():
    validate_contract(load_dataset("test", verify=False))  # must not raise


def test_negative_count_rejected():
    df = load_dataset("train", verify=False).copy()
    df.loc[0, "cnt_user_engagement"] = -1
    with pytest.raises(AssertionError):
        validate_contract(df)


def test_bad_label_rejected():
    df = load_dataset("train", verify=False).copy()
    df.loc[0, "churned"] = 2
    with pytest.raises(AssertionError):
        validate_contract(df)


def test_excluded_rows_rejected():
    # spec §2: bounced and is_enable=0 rows are excluded upstream (0% present).
    df = load_dataset("train", verify=False).copy()
    df.loc[0, "bounced"] = 1
    with pytest.raises(AssertionError):
        validate_contract(df)


def test_out_of_range_prevalence_rejected():
    df = load_dataset("train", verify=False).copy()
    df["churned"] = 0  # prevalence 0 -> outside [0.20, 0.26]
    with pytest.raises(AssertionError):
        validate_contract(df)
