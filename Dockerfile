# Dockerfile simple single-stage para el pipeline de churn.
FROM python:3.12-slim

WORKDIR /app

# Copiar metadatos e instalar el paquete con sus dependencias.
COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir .

# Datos y directorio de artefactos.
COPY data/ data/
RUN mkdir -p artifacts

# Por defecto expone el CLI de entrenamiento.
ENTRYPOINT ["python", "-m", "src.train"]
CMD ["train", "--help"]
