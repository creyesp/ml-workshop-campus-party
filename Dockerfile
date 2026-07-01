# Imagen base oficial de Python (slim) para producción — etapa única
FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python primero (aprovecha caché de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar sólo el código fuente
COPY src/ ./src/

# Variable de entorno para importaciones correctas del paquete
ENV PYTHONPATH=/app

# Los datos se montan en tiempo de ejecución con -v, no se copian en la imagen.
# Ejemplo: docker run -v $(pwd)/data:/app/data -v $(pwd)/artifacts:/app/artifacts ...
VOLUME ["/app/data", "/app/artifacts", "/app/models"]

ENTRYPOINT ["python", "-m", "src.cli"]
