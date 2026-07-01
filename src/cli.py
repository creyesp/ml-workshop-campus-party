import argparse
import json
import sys

from src.config.settings import (
    DEFAULT_TRAIN_PARAMS,
    HPO_DEFAULT_N_TRIALS,
    MODEL_REGISTRY,
)
from src.data.source import load_train_data
from src.train.hpo import run_hpo
from src.train.pipeline import run_train
from src.train.registry import list_runs


def _parse_params(param_str: str | None) -> dict | None:
    if param_str is None:
        return None
    params = {}
    for item in param_str.split(","):
        key, value = item.split("=")
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        params[key.strip()] = value
    return params


def cmd_train(args):
    model_params = _parse_params(args.params) or DEFAULT_TRAIN_PARAMS.get(
        args.model, {}
    )
    result = run_train(
        model_name=args.model,
        model_params=model_params,
        most_common_thr=args.most_common_thr,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_compare(args):
    runs = list_runs()
    if not runs:
        print("No hay corridas disponibles en artifacts/.", file=sys.stderr)
        return
    comparison = []
    for run in runs:
        test_metrics = run.get("metrics", {}).get("test", {})
        comparison.append(
            {
                "run_id": run["run_id"],
                "model_name": run["model_name"],
                "roc_auc": test_metrics.get("roc_auc"),
                "f1": test_metrics.get("f1"),
                "precision": test_metrics.get("precision"),
                "recall": test_metrics.get("recall"),
                "accuracy": test_metrics.get("accuracy"),
            }
        )
    comparison.sort(key=lambda x: x.get("roc_auc", 0) or 0, reverse=True)
    output = {"comparison": comparison}
    with open("comparison.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print("\nResultados guardados en comparison.json", file=sys.stderr)


def cmd_hpo(args):
    x, y = load_train_data()
    study = run_hpo(
        x=x,
        y=y,
        model_name=args.model,
        n_trials=args.n_trials or HPO_DEFAULT_N_TRIALS,
    )
    result = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_predict(args):
    import pandas as pd

    from src.train.registry import load_artifact

    pipeline, metadata = load_artifact(args.run_dir)
    df = pd.read_csv(args.input)
    if args.proba:
        preds = pipeline.predict_proba(df)[:, 1]
    else:
        preds = pipeline.predict(df)
    output = pd.DataFrame({"prediction": preds})
    output.to_csv(args.output, index=False)
    print(
        f"Predicciones guardadas en {args.output} ({len(output)} registros)",
        file=sys.stderr,
    )


def cmd_list(args):
    runs = list_runs()
    if not runs:
        print("No hay corridas registradas.")
        return
    for run in runs:
        test_metrics = run.get("metrics", {}).get("test", {})
        print(
            f"{run['run_id']:16s}  {run['model_name']:30s}  "
            f"roc_auc={test_metrics.get('roc_auc', 'N/A'):>8}"
        )


def main():
    parser = argparse.ArgumentParser(description="ML Pipeline CLI - Churn Prediction")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Entrenar un modelo")
    train_parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Nombre del modelo a entrenar",
    )
    train_parser.add_argument(
        "--params",
        default=None,
        help="Hiperparámetros en formato clave=valor,clave2=valor2",
    )
    train_parser.add_argument(
        "--most-common-thr",
        type=float,
        default=None,
        help="Umbral para agrupar categorías poco frecuentes (0-1)",
    )
    train_parser.set_defaults(func=cmd_train)

    compare_parser = subparsers.add_parser(
        "compare", help="Comparar todas las corridas registradas"
    )
    compare_parser.set_defaults(func=cmd_compare)

    hpo_parser = subparsers.add_parser("hpo", help="Optimización de hiperparámetros")
    hpo_parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Nombre del modelo a optimizar",
    )
    hpo_parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Número de trials para HPO",
    )
    hpo_parser.set_defaults(func=cmd_hpo)

    predict_parser = subparsers.add_parser(
        "predict", help="Inferencia con un modelo entrenado"
    )
    predict_parser.add_argument(
        "--run-dir", required=True, help="Directorio del artifact (run_id)"
    )
    predict_parser.add_argument("--input", required=True, help="CSV de entrada")
    predict_parser.add_argument(
        "--output",
        default="predictions.csv",
        help="CSV de salida con predicciones",
    )
    predict_parser.add_argument(
        "--proba",
        action="store_true",
        help="Predecir probabilidades en lugar de clases",
    )
    predict_parser.set_defaults(func=cmd_predict)

    list_parser = subparsers.add_parser("list", help="Listar corridas")
    list_parser.set_defaults(func=cmd_list)

    parsed = parser.parse_args()
    parsed.func(parsed)


if __name__ == "__main__":
    main()
