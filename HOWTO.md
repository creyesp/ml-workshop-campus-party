# HOWTO — Comandos operativos

## Requisitos

- Python 3.9+
- `uv` (recomendado) o `pip`

## Instalación

```bash
uv sync
```

## Entrenamiento

```bash
# Entrenar una regresión logística
python -m src train --model LogisticRegression

# Entrenar un Random Forest
python -m src train --model RandomForestClassifier

# Entrenar Gradient Boosting
python -m src train --model GradientBoostingClassifier

# Entrenar XGBoost
python -m src train --model XGBClassifier

# Con parámetros personalizados
python -m src train --model RandomForestClassifier --params "n_estimators=200,max_depth=15"

# Con umbral para agrupar categorías poco frecuentes
python -m src train --model LogisticRegression --most-common-thr 0.85
```

## Comparación de modelos

```bash
python -m src compare
```

Genera `comparison.json` con todas las corridas ordenadas por AUC-ROC.

## Optimización de hiperparámetros (HPO)

```bash
python -m src hpo --model XGBClassifier
python -m src hpo --model RandomForestClassifier --n-trials 50
```

## Listar corridas

```bash
python -m src list
```

## Inferencia

```bash
python -m src predict --run-dir artifacts/202501011200 --input data/users_test.csv --output predicciones.csv

# Con probabilidades
python -m src predict --run-dir artifacts/202501011200 --input data/users_test.csv --proba
```

## Docker

```bash
# Construir imagen
docker build -t churn-pipeline .

# Ejecutar entrenamiento
docker run --rm -v $(pwd)/artifacts:/app/artifacts churn-pipeline train --model LogisticRegression

# Ejecutar comparación
docker run --rm -v $(pwd)/artifacts:/app/artifacts churn-pipeline compare
```

## Calidad de código

```bash
uv run ruff check src/
uv run ruff format src/
```

## Tests

Los tests se incorporarán en una fase posterior. La estructura en `tests/` está preparada para pytest.
