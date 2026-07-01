# ML Churn Pipeline

Pipeline de Machine Learning modular, profesional y ejecutable por CLI para predicción de churn de usuarios, construido a partir de notebooks exploratorios.

## Estructura del Proyecto

```
.
├── src/                        # Paquete Python de producción
│   ├── __main__.py             # Punto de entrada para `python -m src`
│   ├── cli.py                  # CLI unificada (train / compare / hpo / predict)
│   ├── config/
│   │   └── settings.py         # Config centralizada: columnas, rutas, hiperparámetros
│   ├── data/
│   │   ├── source.py           # Capa desacoplada de lectura (CSV / BigQuery)
│   │   └── processing.py       # Limpieza, reglas de negocio, split train/test
│   ├── features/
│   │   └── build.py            # Pipeline sklearn: ColumnTransformer + MostCommonCategories
│   └── train/
│       ├── training.py         # Fábrica de modelos + ProductionInferenceWrapper
│       ├── metrics.py          # Funciones puras de métricas (clasificación y regresión)
│       ├── evaluation.py       # Evaluación y gráficos de diagnóstico
│       ├── hpo.py              # Ajuste de hiperparámetros con Optuna
│       ├── registry.py         # Registro de artefactos versionados por corrida
│       └── pipeline.py         # Orquestador (sin lógica de negocio interna)
├── notebooks/                  # Exploración y análisis (no operativos)
├── data/                       # Datos crudos y splits (no versionados en git)
├── artifacts/                  # Artefactos por corrida: modelo, métricas, plots (no versionados)
├── models/                     # Modelos "latest" por tipo (no versionados)
├── tests/                      # Suite de tests unitarios e integración
├── Dockerfile                  # Imagen single-stage para producción
├── .dockerignore
├── pyproject.toml              # Dependencias y configuración de herramientas
├── HOWTO.md                    # Guía completa de uso de la CLI y Docker
└── requirements.txt            # Dependencias para instalación con pip
```

## Instalación

### Con `uv` (recomendado)

```bash
uv sync
```

### Con `pip`

```bash
pip install -r requirements.txt
```

## Uso por CLI

Todos los comandos se ejecutan con `python -m src` o `python -m src.cli`:

### Entrenar un modelo

```bash
python -m src train --model xgb --task classification --data data/users_train.csv
```

Modelos disponibles: `linear`, `rf`, `gb`, `xgb`, `quantile`.  
Tareas: `classification` (default), `regression`.

### Comparar familias de modelos

```bash
python -m src compare --models linear rf xgb --data data/users_train.csv
# → genera artifacts/comparison.json
```

### Ajuste de hiperparámetros (HPO)

```bash
python -m src hpo --model xgb --n-trials 20 --data data/users_train.csv
# → genera artifacts/hpo_xgb_<timestamp>.json
```

### Inferencia en producción

```bash
# Con el modelo más reciente de un tipo
python -m src predict --data data/users_test.csv --output data/predictions.csv --model-type xgb

# Con una corrida específica
python -m src predict --data data/users_test.csv --output data/predictions.csv --run-id 202607010336_xgb
```

Ver la guía completa de operación en [HOWTO.md](HOWTO.md).

## Tests

```bash
uv run pytest tests/ -v
```

## Calidad de Código

```bash
uv run ruff check src/
uv run ruff format src/
```

## Docker

```bash
# Build
docker build -t ml-churn-pipeline:latest .

# Entrenar montando datos locales
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/artifacts:/app/artifacts \
  ml-churn-pipeline:latest train --model xgb --data data/users_train.csv
```

## Artefactos por Corrida

Cada entrenamiento escribe en `artifacts/<YYYYMMDDHHMM>_<model>/`:

```
artifacts/202607010336_xgb/
├── model.joblib          # Pipeline serializado completo
├── metrics.json          # Métricas de evaluación
├── metadata.json         # Columnas, parámetros, estadísticas del split
└── evaluation_plots.png  # Curvas ROC, PR, matriz de confusión, distribución
```
