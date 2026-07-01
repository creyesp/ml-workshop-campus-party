"""
Capa de datos desacoplada.
Contiene la lógica para leer datos de diferentes fuentes (CSV local, BigQuery, etc.).
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd

# Evitar fallos de importación si google-cloud-bigquery no está instalado
try:
    from google.cloud.bigquery import Client
except ImportError:
    Client = None


def load_from_csv(file_path: str) -> pd.DataFrame:
    """
    Carga un conjunto de datos desde un archivo CSV local.

    Args:
        file_path (str): Ruta al archivo CSV.

    Returns:
        pd.DataFrame: Dataframe de Pandas con los datos cargados.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo CSV no existe en la ruta: {file_path}")
    return pd.read_csv(file_path)


def load_from_bigquery(query_path: str, project_id: str) -> pd.DataFrame:
    """
    Ejecuta una consulta SQL en BigQuery y descarga los resultados como DataFrame.

    Args:
        query_path (str): Ruta al archivo .sql que contiene la consulta.
        project_id (str): Identificador del proyecto de Google Cloud.

    Returns:
        pd.DataFrame: Dataframe de Pandas con los datos de la consulta.
    """
    if Client is None:
        raise ImportError("La librería 'google-cloud-bigquery' no está disponible en este entorno.")

    if not os.path.exists(query_path):
        raise FileNotFoundError(f"El archivo de consulta no existe en: {query_path}")

    client = Client(project=project_id)
    with open(query_path, encoding="utf-8") as f:
        query = f.read()

    # Ejecutar query y convertir a dataframe
    query_job = client.query(query)
    dataset = query_job.to_dataframe(create_bqstorage_client=False)
    return dataset


def get_data(source_uri: str, project_id: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """
    Punto de entrada genérico para cargar datos. Soporta archivos locales
    y consultas a BigQuery según el formato de source_uri.

    Args:
        source_uri (str): Ruta a un CSV local o ruta a un archivo .sql.
        project_id (str, opcional): ID del proyecto GCP si se lee de BigQuery.

    Returns:
        pd.DataFrame: Dataframe cargado.
    """
    path = Path(source_uri)
    if path.suffix == ".sql":
        if not project_id:
            raise ValueError("Se requiere 'project_id' para realizar consultas en BigQuery.")
        return load_from_bigquery(source_uri, project_id)
    elif path.suffix == ".csv" or ".csv" in source_uri:
        return load_from_csv(source_uri)
    else:
        raise ValueError(
            f"Formato de URI de origen no soportado: {source_uri}. Debe ser .csv o .sql"
        )
