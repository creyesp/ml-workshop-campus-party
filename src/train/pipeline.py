"""
Orquestador principal del pipeline de entrenamiento (src/train/pipeline.py).
Coordina las etapas de carga, procesamiento, entrenamiento, evaluación y registro,
manteniéndose libre de lógica de negocio o de modelado interno.
"""

import datetime
import json
from pathlib import Path
from typing import Any, Optional

from src.config.settings import (
    ARTIFACTS_DIR,
    CATEGORICAL_COLUMNS,
    DEFAULT_TEST_SIZE,
    LABEL_COLUMN,
    NUMERICAL_COLUMNS,
)
from src.data.processing import (
    filter_by_business_rules,
    sanitize_raw_data,
    split_dataset,
)
from src.data.source import get_data
from src.train.evaluation import (
    evaluate_classification,
    evaluate_regression,
    save_classification_plots,
    save_regression_plots,
)
from src.train.hpo import run_hpo_study
from src.train.registry import generate_run_id, save_run_artifacts
from src.train.training import create_training_pipeline


def run_training_pipeline(
    model_name: str,
    task: str = "classification",
    data_path: str = "",
    test_size: float = DEFAULT_TEST_SIZE,
    use_most_common: bool = False,
    hyperparameters: Optional[dict[str, Any]] = None,
) -> tuple[str, dict[str, Any]]:
    """
    Orquesta la ejecución completa del flujo de entrenamiento y evaluación.

    Args:
        model_name (str): Nombre del modelo ('linear', 'rf', 'gb', 'xgb').
        task (str): Tarea ('classification' o 'regression').
        data_path (str): Ruta al archivo de datos de origen (CSV o SQL).
        test_size (float): Tamaño del split de testeo.
        use_most_common (bool): Reducir cardinalidad en categóricas.
        hyperparameters (Dict, opcional): Hiperparámetros del modelo.

    Returns:
        Tuple[str, Dict[str, Any]]: El run_id generado y el diccionario de métricas.
    """
    # 1. Lectura de datos
    df_raw = get_data(data_path)

    # 2. Sanitización y limpieza
    df_sanitized = sanitize_raw_data(df_raw)

    # 3. Aplicar reglas de negocio para clasificar datos aptos
    df_eligible, df_not_yet = filter_by_business_rules(df_sanitized)

    # Si hay registros 'not yet', guardarlos por separado (simulación de pipeline de negocio)
    if len(df_not_yet) > 0:
        not_yet_path = Path(data_path).parent / "users_not_yet.csv"
        df_not_yet.to_csv(not_yet_path, index=False)

    # 4. Dividir dataset
    train_df, test_df = split_dataset(df_eligible, test_size=test_size)

    # 5. Crear el pipeline estructurado
    model_wrapper = create_training_pipeline(
        model_name=model_name,
        task=task,
        hyperparameters=hyperparameters,
        use_most_common=use_most_common,
    )

    # Separar variables predictoras del target
    x_train = train_df.drop(columns=[LABEL_COLUMN], errors="ignore")
    y_train = train_df[LABEL_COLUMN]
    x_test = test_df.drop(columns=[LABEL_COLUMN], errors="ignore")
    y_test = test_df[LABEL_COLUMN]

    # 6. Entrenamiento (fit) del modelo completo
    model_wrapper.fit(x_train, y_train)

    # 7. Inferencia sobre el conjunto de test
    predictions_df = model_wrapper.predict(x_test)

    # 8. Evaluación y generación de métricas según la tarea
    temp_plot_path = f"/tmp/eval_plots_{model_name}.png"

    if task == "classification":
        y_prob = model_wrapper.predict_proba(x_test)[:, 1]
        metrics_dict = evaluate_classification(
            y_true=y_test.to_numpy(),
            y_pred=predictions_df["prediction"].to_numpy(),
            y_prob=y_prob,
        )
        # Generar gráficos de clasificación
        save_classification_plots(
            y_true=y_test.to_numpy(),
            y_pred=predictions_df["prediction"].to_numpy(),
            y_prob=y_prob,
            output_path=temp_plot_path,
        )
    else:
        # Tarea de regresión
        # Si es de cuantiles, evaluar sobre la predicción del cuantil 0.5 (mediana)
        if "quantile_50" in predictions_df.columns:
            y_pred = predictions_df["quantile_50"].to_numpy()
        else:
            y_pred = predictions_df["prediction"].to_numpy()

        metrics_dict = evaluate_regression(
            y_true=y_test.to_numpy(),
            y_pred=y_pred,
        )
        # Generar gráficos de regresión
        save_regression_plots(
            y_true=y_test.to_numpy(),
            y_pred=y_pred,
            output_path=temp_plot_path,
        )

    # 9. Registro de artefactos
    run_id = generate_run_id(model_name)

    metadata = {
        "model_name": model_name,
        "task": task,
        "features": {
            "numerical": NUMERICAL_COLUMNS,
            "categorical": CATEGORICAL_COLUMNS,
        },
        "data_statistics": {
            "train_samples": len(train_df),
            "test_samples": len(test_df),
        },
        "hyperparameters": hyperparameters or {},
        "use_most_common": use_most_common,
    }

    save_run_artifacts(
        run_id=run_id,
        model=model_wrapper,
        metrics_dict=metrics_dict,
        metadata_dict=metadata,
        plot_file_path=temp_plot_path,
    )

    return run_id, metrics_dict


def run_hpo_pipeline(
    model_name: str,
    task: str = "classification",
    data_path: str = "",
    n_trials: int = 10,
    timeout: int = 600,
    output_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Orquesta la ejecución de la búsqueda de hiperparámetros y persiste
    los resultados en disco como artefacto JSON.

    Args:
        model_name (str): Nombre del modelo ('xgb', 'rf', 'linear').
        task (str): Tarea ('classification' o 'regression').
        data_path (str): Ruta a los datos de entrenamiento.
        n_trials (int): Número de iteraciones.
        timeout (int): Límite de tiempo en segundos.
        output_path (str, opcional): Ruta donde guardar el JSON. Si es None,
            se escribe en artifacts/hpo_{model_name}_{timestamp}.json.

    Returns:
        Dict[str, Any]: Diccionario con los mejores parámetros e info del estudio.
    """
    df_raw = get_data(data_path)
    df_sanitized = sanitize_raw_data(df_raw)
    df_eligible, _ = filter_by_business_rules(df_sanitized)

    # HPO sólo utiliza los datos elegibles de entrenamiento.
    # El conjunto de test queda completamente aislado.
    train_df, _ = split_dataset(df_eligible, test_size=DEFAULT_TEST_SIZE)

    hpo_results = run_hpo_study(
        train_df=train_df,
        model_name=model_name,
        task=task,
        n_trials=n_trials,
        timeout=timeout,
    )

    # Enriquecer el resultado con metadatos de la ejecución
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
    hpo_results["model_name"] = model_name
    hpo_results["task"] = task
    hpo_results["timestamp"] = timestamp
    hpo_results["n_trials_requested"] = n_trials

    # Determinar ruta de salida y persistir
    if output_path is None:
        artifact_path = ARTIFACTS_DIR / f"hpo_{model_name}_{timestamp}.json"
    else:
        artifact_path = Path(output_path)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(hpo_results, f, indent=4, ensure_ascii=False)

    print(f"Resultados HPO guardados en: {artifact_path}")
    return hpo_results


def run_comparison_pipeline(
    models: list[str],
    task: str = "classification",
    data_path: str = "",
    test_size: float = DEFAULT_TEST_SIZE,
) -> dict[str, Any]:
    """
    Entrena y compara múltiples familias de modelos sobre el mismo conjunto de datos,
    y guarda un archivo comparison.json estruturado con la comparación de resultados.

    Args:
        models (List[str]): Lista de modelos a comparar (ej. ['linear', 'rf', 'xgb']).
        task (str): Tarea de ML.
        data_path (str): Ruta a los datos.
        test_size (float): Proporción del split.

    Returns:
        Dict[str, Any]: Resultados de la comparación.
    """
    comparison_results = {}

    for model_name in models:
        try:
            print(f"Entrenando modelo para comparación: {model_name}...")
            run_id, metrics = run_training_pipeline(
                model_name=model_name,
                task=task,
                data_path=data_path,
                test_size=test_size,
            )
            comparison_results[model_name] = {
                "run_id": run_id,
                "metrics": metrics,
            }
        except Exception as e:
            print(f"Error entrenando modelo {model_name} en la comparación: {e}")
            comparison_results[model_name] = {"error": str(e)}

    # Guardar la comparación estructurada en artifacts/
    comparison_path = ARTIFACTS_DIR / "comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=4, ensure_ascii=False)

    print(f"Resultados de comparación guardados en: {comparison_path}")
    return comparison_results
