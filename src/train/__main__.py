"""
CLI de entrenamiento, comparación, HPO e inferencia.

Uso:
    python -m src.train train   [--model M] [--family F] [--params k=v,...] [--no-save]
    python -m src.train compare [--models M1 M2 ...] [--family F]
    python -m src.train hpo     [--model M] [--trials N]
    python -m src.train predict --run-id RUN_ID --input datos.csv [--output out.csv]

Toda la lógica vive en los módulos del paquete; aquí solo se parsean argumentos y se
invocan los orquestadores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config.settings import get_settings
from src.train.compare import compare_models
from src.train.hpo import run_hpo
from src.train.inference import load_model
from src.train.pipeline import run_training


def _parse_params(raw: str | None) -> dict:
    """Convierte ``k=v,k=v`` en un dict con valores numéricos cuando aplica."""
    if not raw:
        return {}
    params: dict[str, object] = {}
    for pair in raw.split(","):
        key, _, value = pair.partition("=")
        key, value = key.strip(), value.strip()
        try:
            number = float(value)
            params[key] = int(number) if number.is_integer() else number
        except ValueError:
            params[key] = value
    return params


def _cmd_train(args: argparse.Namespace) -> None:
    result = run_training(
        model_name=args.model,
        family=args.family,
        model_params=_parse_params(args.params),
        save=not args.no_save,
    )
    print(
        json.dumps(
            {"run_id": result["run_id"], "metrics": result["metrics"]["test"]}, indent=2
        )
    )


def _cmd_compare(args: argparse.Namespace) -> None:
    comparison = compare_models(model_names=args.models, family=args.family)
    print(json.dumps(comparison, indent=2))


def _cmd_hpo(args: argparse.Namespace) -> None:
    result = run_hpo(model_name=args.model, n_trials=args.trials)
    print(json.dumps(result, indent=2))


def _cmd_predict(args: argparse.Namespace) -> None:
    run_dir = get_settings().artifacts.artifacts_dir / args.run_id
    model_path = run_dir / get_settings().artifacts.model_filename
    model = load_model(model_path)
    dataframe = pd.read_csv(args.input)
    predictions = model.predict(dataframe)
    if args.output:
        predictions.to_csv(args.output, index=False)
        print(f"Predicciones guardadas en {args.output} ({len(predictions)} filas)")
    else:
        print(predictions.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(prog="src.train", description="Pipeline de churn.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Entrenar un modelo.")
    train_parser.add_argument(
        "--model",
        choices=settings.training.available_models,
        default=settings.training.default_model,
    )
    train_parser.add_argument(
        "--family",
        choices=settings.training.available_families,
        default=settings.training.default_family,
    )
    train_parser.add_argument("--params", help="Hiperparámetros k=v,k=v.")
    train_parser.add_argument(
        "--no-save", action="store_true", help="No guardar artefactos."
    )
    train_parser.set_defaults(func=_cmd_train)

    compare_parser = subparsers.add_parser("compare", help="Comparar modelos.")
    compare_parser.add_argument(
        "--models",
        nargs="+",
        choices=settings.training.available_models,
        help="Modelos a comparar (default: todos).",
    )
    compare_parser.add_argument(
        "--family",
        choices=settings.training.available_families,
        default=settings.training.default_family,
    )
    compare_parser.set_defaults(func=_cmd_compare)

    hpo_parser = subparsers.add_parser("hpo", help="Optimizar hiperparámetros.")
    hpo_parser.add_argument(
        "--model",
        choices=settings.training.available_models,
        default=settings.training.default_model,
    )
    hpo_parser.add_argument("--trials", type=int, help="Número de trials de Optuna.")
    hpo_parser.set_defaults(func=_cmd_hpo)

    predict_parser = subparsers.add_parser(
        "predict", help="Inferencia con un artefacto."
    )
    predict_parser.add_argument("--run-id", required=True, help="RUN_ID del modelo.")
    predict_parser.add_argument(
        "--input", required=True, type=Path, help="CSV de entrada."
    )
    predict_parser.add_argument("--output", type=Path, help="CSV de salida (opcional).")
    predict_parser.set_defaults(func=_cmd_predict)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
