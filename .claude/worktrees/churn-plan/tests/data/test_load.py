import pytest

from floodit.data.load import sha256_of, load_dataset


def test_sha256_is_deterministic(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n")
    assert sha256_of(p) == sha256_of(p)
    assert len(sha256_of(p)) == 64


def test_load_train_shape_and_balance():
    df = load_dataset("train", verify=False)
    assert df.shape == (7190, 19)
    assert round(df["churned"].mean(), 3) == 0.231


def test_load_test_shape():
    df = load_dataset("test", verify=False)
    assert df.shape == (799, 19)


def test_first_engagement_parsed_as_datetime():
    df = load_dataset("train", verify=False)
    assert str(df["user_first_engagement"].dtype).startswith("datetime64")


def test_unknown_name_raises():
    with pytest.raises(ValueError):
        load_dataset("nope", verify=False)


def test_verify_true_passes_with_pinned_hash():
    # Pinned digests (Task 2) must match the on-disk files.
    df = load_dataset("train", verify=True)
    assert len(df) == 7190
