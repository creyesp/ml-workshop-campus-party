from floodit_churn.model import (
    evaluate,
    load_metadata,
    load_model,
    save_model,
    train,
)


def test_train_predicts(sample_df):
    model = train(sample_df)
    proba = model.predict_proba(sample_df)[:, 1]
    assert proba.shape[0] == len(sample_df)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_evaluate_returns_metrics(sample_df):
    model = train(sample_df)
    metrics = evaluate(model, sample_df, threshold=0.5)
    for key in ("roc_auc", "pr_auc", "precision", "recall", "f1"):
        assert key in metrics


def test_save_load_roundtrip(sample_df, tmp_path):
    model = train(sample_df)
    out = save_model(model, tmp_path / "m", metadata={"note": "test"})
    loaded = load_model(out)
    meta = load_metadata(out)
    assert meta["note"] == "test"
    assert "sklearn_version" in meta
    # Predicciones idénticas tras el round-trip.
    a = model.predict_proba(sample_df)[:, 1]
    b = loaded.predict_proba(sample_df)[:, 1]
    assert (a == b).all()
