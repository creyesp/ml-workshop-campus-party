"""
Lectura de datos desacoplada de la fuente.

La lógica de lectura queda aislada detrás de la interfaz ``DataSource`` para poder
reemplazar el backend (CSV local, BigQuery, API, data warehouse) sin tocar el resto
del pipeline. Hoy el default es CSV; ``BigQueryDataSource`` queda como punto de
extensión.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from src.config.settings import get_settings


class DataSource(ABC):
    """Interfaz abstracta para una fuente de datos tabular."""

    @abstractmethod
    def read(self) -> pd.DataFrame:
        """Lee la fuente y devuelve un DataFrame con datos crudos."""
        raise NotImplementedError


class CSVDataSource(DataSource):
    """Fuente de datos basada en un archivo CSV local."""

    def __init__(self, file_path: Path | None = None):
        config = get_settings().data
        self.file_path = (
            Path(file_path) if file_path else config.data_dir / config.raw_file
        )

    def read(self) -> pd.DataFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Archivo de datos no encontrado: {self.file_path}")
        dataframe = pd.read_csv(self.file_path)
        if get_settings().verbose:
            print(f"[data] CSV leído desde {self.file_path} -> shape {dataframe.shape}")
        return dataframe


class BigQueryDataSource(DataSource):
    """
    Fuente de datos basada en BigQuery (punto de extensión).

    Ejecuta una query almacenada en un archivo ``.sql``. Requiere el paquete
    ``google-cloud-bigquery`` instalado y credenciales configuradas.
    """

    def __init__(self, project_id: str, query_path: Path):
        self.project_id = project_id
        self.query_path = Path(query_path)

    def read(self) -> pd.DataFrame:
        try:
            from google.cloud.bigquery import Client
        except ImportError as error:  # pragma: no cover - dependencia opcional
            raise ImportError(
                "google-cloud-bigquery no está instalado; instálalo para usar BigQuery."
            ) from error

        if not self.query_path.exists():
            raise FileNotFoundError(f"Query no encontrada: {self.query_path}")

        query = self.query_path.read_text()
        client = Client(project=self.project_id)
        dataframe = client.query(query).to_dataframe(create_bqstorage_client=False)
        if get_settings().verbose:
            print(f"[data] BigQuery ({self.project_id}) -> shape {dataframe.shape}")
        return dataframe


def get_data_source(source_type: str = "csv", **kwargs) -> DataSource:
    """
    Fábrica de fuentes de datos.

    Args:
        source_type: ``"csv"`` o ``"bigquery"``.
        **kwargs: argumentos específicos de la fuente elegida.
    """
    sources: dict[str, type[DataSource]] = {
        "csv": CSVDataSource,
        "bigquery": BigQueryDataSource,
    }
    if source_type not in sources:
        raise ValueError(
            f"Fuente de datos desconocida: {source_type}. Opciones: {list(sources)}"
        )
    return sources[source_type](**kwargs)


def load_dataset(source_type: str = "csv", **kwargs) -> pd.DataFrame:
    """Carga el dataset crudo desde la fuente indicada."""
    return get_data_source(source_type, **kwargs).read()
