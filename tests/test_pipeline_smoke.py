"""Test de humo end-to-end: entrenamiento, evaluación e inferencia."""

from __future__ import annotations

from src.config.settings import get_settings
from src.data.processing import (
    sanitize_dataset,
    split_data,
    split_features_target,
)
from src.train.evaluation import evaluate_model
from src.train.inference import build_inference_model
from src.train.training import train_model


def test_train_evaluate_predict(sample_dataframe):
    """El flujo completo produce métricas válidas y salida de negocio."""
    clean = sanitize_dataset(sample_dataframe)
    train_df, test_df = split_data(clean)
    x_train, y_train = split_features_target(train_df)
    x_test, y_test = split_features_target(test_df)

    model = train_model(x_train, y_train, model_name="logistic")
    metrics = evaluate_model(model, x_train, y_train, x_test, y_test)

    assert 0.0 <= metrics["test"]["roc_auc"] <= 1.0
    assert set(metrics) == {"train", "test"}

    inference = build_inference_model(model, model_name="logistic", family="point")
    output = inference.predict(x_test)
    assert "churn_probability" in output.columns
    assert "churn_prediction" in output.columns
    assert output["churn_probability"].between(0, 1).all()


def test_default_model_is_available():
    settings = get_settings()
    assert settings.training.default_model in settings.training.available_models
