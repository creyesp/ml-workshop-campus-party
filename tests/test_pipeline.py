"""
Pruebas de integración de extremo a extremo para el pipeline de entrenamiento.
"""

import os
import pandas as pd
from src.train.pipeline import run_training_pipeline
from src.train.registry import load_model_pipeline


def test_end_to_end_pipeline(tmp_path):
    """
    Entrena un modelo rápido de prueba lineal en un dataset temporal pequeño
    y verifica que se cree e infiera correctamente.
    """
    # 1. Crear dataset de prueba mínimo
    data = {
        "churned": [1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
        "is_enable": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "bounced": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "country_name": [
            "US",
            "AR",
            "US",
            "AR",
            "BR",
            "BR",
            "US",
            "AR",
            "US",
            "BR",
        ],
        "device_os": [
            "iOS",
            "Android",
            "iOS",
            "Android",
            "iOS",
            "Android",
            "iOS",
            "Android",
            "iOS",
            "Android",
        ],
        "device_lang": [
            "en",
            "es",
            "en",
            "es",
            "pt",
            "pt",
            "en",
            "es",
            "en",
            "pt",
        ],
        "user_first_engagement": ["2026-01-01"] * 10,
        "user_pseudo_id": [f"user_{i}" for i in range(10)],
    }

    # Agregar columnas numéricas requeridas
    from src.config.settings import NUMERICAL_COLUMNS

    for col in NUMERICAL_COLUMNS:
        data[col] = [1.0, 2.0, 1.5, 3.0, 0.5, 4.0, 1.2, 2.8, 1.9, 0.2]

    df = pd.DataFrame(data)
    temp_csv_path = tmp_path / "mock_users_train.csv"
    df.to_csv(temp_csv_path, index=False)

    # 2. Correr el pipeline de entrenamiento
    run_id, metrics = run_training_pipeline(
        model_name="linear",
        task="classification",
        data_path=str(temp_csv_path),
        test_size=0.3,
        use_most_common=False,
    )

    # 3. Validar resultados y existencia de artefactos
    assert run_id is not None
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics

    # 4. Cargar modelo del registro y validar predicción
    model = load_model_pipeline(run_id)
    predictions_df = model.predict(df)

    assert "prediction" in predictions_df.columns
    assert "probability" in predictions_df.columns
    assert "decision" in predictions_df.columns
    assert len(predictions_df) == 10
