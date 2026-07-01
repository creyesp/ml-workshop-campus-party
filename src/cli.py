"""
Interfaz de línea de comandos (CLI) principal para interactuar con el pipeline.
Admite subcomandos para entrenamiento, comparación, optimización (HPO) e inferencia.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from src.config.settings import (
    DEFAULT_TEST_SIZE,
    DEFAULT_TRAIN_DATA_PATH,
    HPO_NUM_TRIALS,
    HPO_TIMEOUT_SECONDS,
)
from src.data.source import get_data
from src.train.pipeline import (
    run_comparison_pipeline,
    run_hpo_pipeline,
    run_training_pipeline,
)
from src.train.registry import load_latest_model, load_model_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="CLI para el ciclo de vida modular de Machine Learning (Producción)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcomandos disponibles")

    # --- SUBCOMANDO: train ---
    train_parser = subparsers.add_parser("train", help="Entrena un modelo individual")
    train_parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["linear", "rf", "gb", "xgb", "quantile"],
        help="Algoritmo de modelo a entrenar",
    )
    train_parser.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=["classification", "regression"],
        help="Tipo de tarea de machine learning",
    )
    train_parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_TRAIN_DATA_PATH),
        help="Ruta del CSV de datos de entrenamiento",
    )
    train_parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Proporción de split para test",
    )
    train_parser.add_argument(
        "--use-most-common",
        action="store_true",
        help="Si se activa, reduce cardinalidad de variables categóricas",
    )
    train_parser.add_argument(
        "--hyperparameters",
        type=str,
        default=None,
        help="JSON string con hiperparámetros personalizados para el modelo",
    )

    # --- SUBCOMANDO: compare ---
    compare_parser = subparsers.add_parser("compare", help="Compara múltiples modelos")
    compare_parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["linear", "rf", "xgb"],
        help="Lista de modelos a entrenar y comparar",
    )
    compare_parser.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=["classification", "regression"],
        help="Tipo de tarea de machine learning",
    )
    compare_parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_TRAIN_DATA_PATH),
        help="Ruta del CSV de datos de entrenamiento",
    )
    compare_parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Proporción de split para test",
    )

    # --- SUBCOMANDO: hpo ---
    hpo_parser = subparsers.add_parser("hpo", help="Ajusta hiperparámetros con Optuna")
    hpo_parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["xgb", "rf", "linear"],
        help="Modelo a optimizar",
    )
    hpo_parser.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=["classification", "regression"],
        help="Tipo de tarea",
    )
    hpo_parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_TRAIN_DATA_PATH),
        help="Ruta de datos",
    )
    hpo_parser.add_argument(
        "--n-trials",
        type=int,
        default=HPO_NUM_TRIALS,
        help="Número de iteraciones",
    )
    hpo_parser.add_argument(
        "--timeout",
        type=int,
        default=HPO_TIMEOUT_SECONDS,
        help="Tiempo de timeout en segundos",
    )
    hpo_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta donde guardar el JSON con los resultados HPO (opcional)",
    )

    # --- SUBCOMANDO: predict (Inferencia) ---
    predict_parser = subparsers.add_parser("predict", help="Ejecuta inferencia (predict)")
    predict_parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Ruta de datos crudos para realizar predicciones",
    )
    predict_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Ruta donde se escribirá el archivo CSV de salida con predicciones de negocio",
    )
    # Cargar usando run_id o por tipo de modelo (cargará el más reciente 'latest')
    group = predict_parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run-id",
        type=str,
        help="ID de corrida específica de donde cargar el modelo",
    )
    group.add_argument(
        "--model-type",
        type=str,
        choices=["linear", "rf", "gb", "xgb", "quantile", "model"],
        help="Carga el modelo marcado como 'latest' de este tipo",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    try:
        if args.command == "train":
            hparams = None
            if args.hyperparameters:
                hparams = json.loads(args.hyperparameters)

            print(f"Iniciando entrenamiento para el modelo '{args.model}'...")
            run_id, metrics = run_training_pipeline(
                model_name=args.model,
                task=args.task,
                data_path=args.data,
                test_size=args.test_size,
                use_most_common=args.use_most_common,
                hyperparameters=hparams,
            )
            print("=========================================")
            print("✓ Entrenamiento exitoso.")
            print(f"RUN_ID: {run_id}")
            print(f"Métricas obtenidas: {json.dumps(metrics, indent=2)}")
            print("=========================================")

        elif args.command == "compare":
            print(f"Iniciando comparación para los modelos: {args.models}...")
            comparison = run_comparison_pipeline(
                models=args.models,
                task=args.task,
                data_path=args.data,
                test_size=args.test_size,
            )
            print("=========================================")
            print("✓ Comparación completada.")
            print(json.dumps(comparison, indent=2))
            print("=========================================")

        elif args.command == "hpo":
            print(f"Iniciando optimización de hiperparámetros para '{args.model}'...")
            hpo_results = run_hpo_pipeline(
                model_name=args.model,
                task=args.task,
                data_path=args.data,
                n_trials=args.n_trials,
                timeout=args.timeout,
                output_path=args.output,
            )
            print("=========================================")
            print("✓ HPO optimización finalizada.")
            print(f"Mejor valor métrica: {hpo_results['best_params']}")
            print(
                f"Hiperparámetros óptimos: {json.dumps(hpo_results['best_hyperparameters'], indent=2)}"
            )
            print("=========================================")

        elif args.command == "predict":
            # Cargar el modelo deseado
            if args.run_id:
                print(f"Cargando modelo de la corrida '{args.run_id}'...")
                model = load_model_pipeline(args.run_id)
            else:
                print(f"Cargando modelo 'latest' para tipo '{args.model_type}'...")
                model = load_latest_model(args.model_type)

            # Cargar datos de entrada
            print(f"Cargando datos desde '{args.data}'...")
            df_input = get_data(args.data)

            # Realizar predicción (devuelve salida con formato de negocio)
            print("Ejecutando predicciones de negocio...")
            predictions_df = model.predict(df_input)

            # Guardar predicciones
            output_path = Path(args.output)
            os.makedirs(output_path.parent, exist_ok=True)
            predictions_df.to_csv(output_path, index=True)
            print(f"✓ Resultados guardados exitosamente en: {output_path}")

    except Exception as e:
        print(f"ERROR: Ocurrió un fallo durante la ejecución: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
