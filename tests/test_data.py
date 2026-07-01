import pandas as pd
import pytest

from floodit_churn.data import SchemaError, load_dataset, sha256, split_xy


def test_load_dataset_ok(sample_csv):
    df = load_dataset(sample_csv, require_label=True)
    assert len(df) > 0


def test_load_dataset_missing_features(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(SchemaError):
        load_dataset(path, require_label=False)


def test_load_dataset_missing_label(sample_df, tmp_path):
    path = tmp_path / "nolabel.csv"
    sample_df.drop(columns=["churned"]).to_csv(path, index=False)
    # Sin label es válido para scoring...
    load_dataset(path, require_label=False)
    # ...pero no para entrenamiento.
    with pytest.raises(SchemaError):
        load_dataset(path, require_label=True)


def test_split_xy(sample_df):
    x, y = split_xy(sample_df)
    assert "churned" not in x.columns
    assert len(x) == len(y)


def test_sha256_stable(sample_csv):
    assert sha256(sample_csv) == sha256(sample_csv)
