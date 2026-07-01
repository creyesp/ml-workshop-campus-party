"""
Configuración centralizada del pipeline de churn.

Todos los parámetros globales (columnas, umbrales, tamaños de split, hiperparámetros
por defecto y espacio de búsqueda de HPO) viven aquí. El resto del código no debe
hardcodear estos valores: siempre debe leerlos desde ``get_settings()``.

El problema es de clasificación binaria: predecir si un usuario abandonará la app
(``churned`` = 1) a partir de sus eventos de engagement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Raíz del repositorio (dos niveles arriba de este archivo: src/config/settings.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Hiperparámetros por defecto de cada familia de modelos.
# Nombres explícitos DEFAULT_*_PARAMS para evitar valores mágicos en training.py.
# ``random_state`` se inyecta en tiempo de construcción desde ``Settings.random_seed``.
# ---------------------------------------------------------------------------
DEFAULT_LOGISTIC_PARAMS: dict = {
    "class_weight": "balanced",
    "max_iter": 1000,
}
DEFAULT_RANDOM_FOREST_PARAMS: dict = {
    "n_estimators": 300,
    "class_weight": "balanced",
    "n_jobs": -1,
}
DEFAULT_GRADIENT_BOOSTING_PARAMS: dict = {
    "n_estimators": 200,
    "learning_rate": 0.1,
}
DEFAULT_XGBOOST_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "eval_metric": "logloss",
}


@dataclass
class DataConfig:
    """Configuración de lectura y saneamiento de datos."""

    data_dir: Path = PROJECT_ROOT / "data"
    raw_file: str = "users_raw.csv"

    # Split interno train/test (estratificado por la etiqueta).
    test_size: float = 0.1
    random_state: int = 42

    # Etiqueta objetivo.
    label_column: str = "churned"

    # Identificadores y columnas con potencial fuga de información que se descartan.
    id_columns: list[str] = field(
        default_factory=lambda: [
            "user_pseudo_id",
            "user_first_engagement",
        ]
    )
    leakage_columns: list[str] = field(
        default_factory=lambda: [
            "is_enable",
            "bounced",
        ]
    )

    # Features usadas por el modelo.
    categorical_features: list[str] = field(
        default_factory=lambda: [
            "country_name",
            "device_os",
            "device_lang",
        ]
    )
    numeric_features: list[str] = field(
        default_factory=lambda: [
            "cnt_user_engagement",
            "cnt_level_start_quickplay",
            "cnt_level_end_quickplay",
            "cnt_level_complete_quickplay",
            "cnt_level_reset_quickplay",
            "cnt_post_score",
            "cnt_spend_virtual_currency",
            "cnt_ad_reward",
            "cnt_challenge_a_friend",
            "cnt_completed_5_levels",
            "cnt_use_extra_steps",
        ]
    )

    @property
    def feature_columns(self) -> list[str]:
        """Todas las columnas que ingresan al pipeline de features."""
        return self.numeric_features + self.categorical_features


@dataclass
class FeatureConfig:
    """Configuración del pipeline de features (sklearn)."""

    numeric_imputer_fill_value: float = 0.0
    categorical_imputer_strategy: str = "most_frequent"
    scale_numeric: bool = True
    onehot_handle_unknown: str = "ignore"


@dataclass
class TrainingConfig:
    """Configuración de entrenamiento y familias de modelos."""

    # Algoritmos disponibles (todos clasificadores).
    available_models: list[str] = field(
        default_factory=lambda: [
            "logistic",
            "random_forest",
            "gradient_boosting",
            "xgboost",
        ]
    )
    default_model: str = "xgboost"

    # Familias de salida:
    #   - "point": predicción de clase (churn sí/no).
    #   - "probabilistic": probabilidad de churn (predict_proba).
    available_families: list[str] = field(
        default_factory=lambda: ["point", "probabilistic"]
    )
    default_family: str = "probabilistic"

    # Umbral de decisión para pasar de probabilidad a clase.
    decision_threshold: float = 0.5


@dataclass
class HPOConfig:
    """Configuración de optimización de hiperparámetros (Optuna)."""

    n_trials: int = 30
    cv_folds: int = 5
    # Se optimiza el ROC-AUC promedio de validación cruzada (mayor es mejor).
    metric_name: str = "roc_auc"
    direction: str = "maximize"

    # Espacio de búsqueda por algoritmo. Cada entrada describe un rango para Optuna.
    search_spaces: dict = field(
        default_factory=lambda: {
            "xgboost": {
                "max_depth": {"type": "int", "low": 3, "high": 10},
                "learning_rate": {
                    "type": "float",
                    "low": 1e-3,
                    "high": 0.3,
                    "log": True,
                },
                "n_estimators": {"type": "int", "low": 100, "high": 600, "step": 50},
                "subsample": {"type": "float", "low": 0.5, "high": 1.0},
                "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
                "min_child_weight": {"type": "int", "low": 1, "high": 10},
            },
            "random_forest": {
                "n_estimators": {"type": "int", "low": 100, "high": 600, "step": 50},
                "max_depth": {"type": "int", "low": 3, "high": 20},
                "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
            },
        }
    )


@dataclass
class ArtifactConfig:
    """Configuración de registro de artefactos por corrida."""

    artifacts_dir: Path = PROJECT_ROOT / "artifacts"

    # Formato del RUN_ID: timestamp YYYYMMDDHHMM.
    run_id_format: str = "%Y%m%d%H%M"

    model_filename: str = "model.joblib"
    metrics_filename: str = "metrics.json"
    metadata_filename: str = "metadata.json"
    comparison_filename: str = "comparison.json"


@dataclass
class Settings:
    """Configuración global unificada."""

    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hpo: HPOConfig = field(default_factory=HPOConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)

    random_seed: int = 42
    verbose: bool = True


# Instancia global reutilizable.
settings = Settings()


def get_settings() -> Settings:
    """Devuelve la instancia global de configuración."""
    return settings
