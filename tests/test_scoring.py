import pandas as pd

from floodit_churn.model import save_model, train
from floodit_churn.scoring import score_csv, score_dataframe


def test_score_dataframe_columns(sample_df):
    model = train(sample_df)
    result = score_dataframe(sample_df, model, threshold=0.5)
    assert list(result.columns) == ["user_pseudo_id", "churn_proba", "churn_pred"]
    assert set(result["churn_pred"].unique()) <= {0, 1}
    assert len(result) == len(sample_df)


def test_threshold_changes_predictions(sample_df):
    model = train(sample_df)
    low = score_dataframe(sample_df, model, threshold=0.1)["churn_pred"].sum()
    high = score_dataframe(sample_df, model, threshold=0.9)["churn_pred"].sum()
    # Umbral más bajo => al menos tantas predicciones positivas.
    assert low >= high


def test_score_csv_writes_output(sample_df, tmp_path):
    model = train(sample_df)
    model_dir = save_model(model, tmp_path / "m", metadata={"default_threshold": 0.5})
    in_csv = tmp_path / "in.csv"
    sample_df.to_csv(in_csv, index=False)
    out_csv = tmp_path / "out" / "preds.csv"

    score_csv(in_csv, model_dir, out_csv)

    assert out_csv.exists()
    preds = pd.read_csv(out_csv)
    assert "churn_proba" in preds.columns
    assert len(preds) == len(sample_df)
