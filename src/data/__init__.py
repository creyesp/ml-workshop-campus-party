"""Capa de datos: lectura desacoplada y saneamiento."""

from src.data.processing import sanitize_dataset, split_data
from src.data.source import load_dataset

__all__ = ["load_dataset", "sanitize_dataset", "split_data"]
