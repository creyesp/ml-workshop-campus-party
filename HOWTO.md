# HOWTO — Pipeline de churn en producción

Guía operativa del paquete `src/`, transformado desde los notebooks del workshop a
código modular ejecutable por CLI. El problema es **clasificación binaria**: predecir
si un usuario abandonará la app (`churned = 1`) a partir de sus eventos de engagement.

## Setup

```bash
# Instalar el paquete y las dependencias de desarrollo (ruff, pytest)
pip install -e '.[dev]'
```

Estructura del paquete:

```
src/
  config/settings.py     # configuración centralizada (columnas, splits, HPO, defaults)
  data/source.py         # lectura desacoplada (CSV / BigQuery)
  data/processing.py     # saneamiento + split estratificado
  features/build.py      # pipeline de features (ColumnTransformer)
  train/training.py      # construcción y fit de clasificadores
  train/metrics.py       # métricas puras (point + probabilistic)
  train/evaluation.py    # evaluación train/test
  train/inference.py     # artefacto de inferencia (features + modelo)
  train/registry.py      # registro de artefactos por corrida
  train/pipeline.py      # orquestador (solo coordina etapas)
  train/compare.py       # comparación de modelos -> comparison.json
  train/hpo.py           # optimización de hiperparámetros (Optuna)
  train/__main__.py      # CLI: train / compare / hpo / predict
```

## Comandos

### Entrenamiento

```bash
# Modelo por defecto (xgboost), familia probabilística
python -m src.train train

# Elegir algoritmo y familia de salida
python -m src.train train --model logistic --family point
python -m src.train train --model random_forest
python -m src.train train --model gradient_boosting

# Hiperparámetros a medida
python -m src.train train --model xgboost --params max_depth=8,learning_rate=0.05

# Entrenar sin guardar artefactos
python -m src.train train --model xgboost --no-save
```

**Modelos disponibles:** `logistic`, `random_forest`, `gradient_boosting`, `xgboost`
(todos clasificadores, con balanceo de clases).

**Familias de salida:**
- `point` — predicción de clase (churn sí/no).
- `probabilistic` — probabilidad de churn (`predict_proba`).

Cada corrida escribe en `artifacts/<RUN_ID>/` (RUN_ID = timestamp `YYYYMMDDHHMM`):
`model.joblib`, `metrics.json`, `metadata.json`.

### Comparación de modelos

```bash
# Comparar todos los modelos disponibles
python -m src.train compare

# Comparar un subconjunto
python -m src.train compare --models logistic xgboost
```

Genera `artifacts/<RUN_ID>/comparison.json` con el ranking por ROC-AUC de test.

### Optimización de hiperparámetros (HPO)

```bash
python -m src.train hpo --model xgboost --trials 30
# equivalente:
python -m src.train.hpo --model xgboost --trials 30
```

El estudio usa validación cruzada estratificada **solo sobre train** (no toca el test
final). El espacio de búsqueda y el número de trials viven en `settings.py`.

### Inferencia

```bash
python -m src.train predict --run-id 202607011230 --input data/users_test.csv --output preds.csv
```

Carga el artefacto de la corrida y devuelve salida de negocio (`churn_probability` y,
en la familia `point`, `churn_prediction`). El pipeline de features viaja dentro del
artefacto: no hay preprocesamiento manual en inferencia.

## Docker

```bash
docker build -t churn-ml .

# Entrenar dentro del contenedor
docker run --rm -v "$PWD/artifacts:/app/artifacts" churn-ml train --model xgboost

# Comparar
docker run --rm -v "$PWD/artifacts:/app/artifacts" churn-ml compare
```

## Calidad de código

```bash
ruff check src tests     # linting
ruff format src tests    # formato
pytest -q                # tests
```

## Tests

La estructura de tests está incluida (`tests/`) con:
- `test_features.py` — el pipeline se ajusta con train y maneja categorías nuevas.
- `test_metrics.py` — corrección de las métricas puras.
- `test_pipeline_smoke.py` — flujo end-to-end (entrenar → evaluar → predecir).

Ampliar la cobertura (data sources, registry, HPO) queda para la fase siguiente.
