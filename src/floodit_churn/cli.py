"""Entrypoints de línea de comandos: floodit-train y floodit-score."""
from __future__ import annotations

import argparse
import json

from .config import ARTIFACTS_DIR, DEFAULT_THRESHOLD, TEST_CSV, TRAIN_CSV
from .model import load_metadata, train_and_freeze
from .scoring import score_csv


def train_cmd(argv: list[str] | None = None) -> None:
    """Reentrena el XGBoost ganador y congela el artefacto con su metadata."""
    parser = argparse.ArgumentParser(prog="floodit-train", description=train_cmd.__doc__)
    parser.add_argument("--data", default=str(TRAIN_CSV), help="CSV de entrenamiento")
    parser.add_argument("--test", default=str(TEST_CSV), help="CSV de test para métricas")
    parser.add_argument(
        "--out", default=str(ARTIFACTS_DIR / "model_v1"), help="directorio del artefacto"
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args(argv)

    out_dir = train_and_freeze(
        train_csv=args.data,
        out_dir=args.out,
        test_csv=args.test,
        threshold=args.threshold,
    )
    meta = load_metadata(out_dir)
    print(f"Modelo congelado en: {out_dir}")
    if "test_metrics" in meta:
        print("Métricas en test:")
        print(json.dumps(meta["test_metrics"], indent=2))


def score_cmd(argv: list[str] | None = None) -> None:
    """Batch scoring de un CSV de usuarios a un CSV local de predicciones."""
    parser = argparse.ArgumentParser(prog="floodit-score", description=score_cmd.__doc__)
    parser.add_argument("--input", required=True, help="CSV de usuarios a puntuar")
    parser.add_argument(
        "--model", default=str(ARTIFACTS_DIR / "model_v1"), help="directorio del artefacto"
    )
    parser.add_argument("--output", required=True, help="CSV de salida con predicciones")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="umbral de decisión (default: el de la metadata, si no 0.5)",
    )
    args = parser.parse_args(argv)

    out = score_csv(
        input_csv=args.input,
        model_dir=args.model,
        output_csv=args.output,
        threshold=args.threshold,
    )
    print(f"Predicciones escritas en: {out}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit("Usa los comandos floodit-train o floodit-score")
