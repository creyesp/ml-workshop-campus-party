# Guía de Operación y Uso (HOWTO)

Esta guía explica cómo ejecutar, entrenar, optimizar e inferir utilizando el paquete modular de Machine Learning de producción en `src/`.

---

## Estructura del Proyecto

El código está estructurado bajo principios de modularidad estricta y desacoplamiento de capas:

- `src/config/settings.py`: Configuración centralizada (hiperparámetros, columnas, rutas, configuraciones de HPO).
- `src/data/source.py`: Capa de adquisición de datos desacoplada (lectura de CSV local o consultas a BigQuery).
- `src/data/processing.py`: Limpieza de datos y división train/test respetando las reglas de negocio.
- `src/features/build.py`: Pipeline de preprocesamiento de características (`ColumnTransformer` y transformador personalizado `MostCommonCategories`).
- `src/train/training.py`: Construcción de modelos con transformaciones target encapsuladas (`TransformedTargetRegressor`) y regresores por cuantiles (`QuantileRegressorWrapper`).
- `src/train/metrics.py`: Funciones puras para métricas puntuales y probabilísticas de clasificación y regresión.
- `src/train/evaluation.py`: Evaluación final del modelo y guardado automático de gráficos de diagnóstico.
- `src/train/registry.py`: Registro y versionado de modelos, métricas y metadatos bajo `artifacts/<RUN_ID>/`.
- `src/train/pipeline.py`: Coordinador central libre de lógica de negocio o detalles internos de ajuste.
- `src/cli.py`: Interfaz de línea de comandos (CLI) unificada.

---

## Ejecución Mediante la CLI

Todas las tareas operativas se ejecutan llamando al módulo principal de la aplicación:

```bash
uv run python -m src.cli [SUBCOMANDO] [OPCIONES]
```

*(Si no utilizas `uv`, puedes usar `python -m src.cli ...` asegurando que tu entorno virtual tenga activas las dependencias de `requirements.txt`).*

### 1. Entrenamiento Individual (`train`)

Permite entrenar un modelo individual seleccionando el algoritmo, tarea y conjunto de datos.

#### Clasificación (Churn)
Entrena un clasificador XGBoost con balanceo de clases:
```bash
uv run python -m src.cli train --model xgb --task classification --data data/users_train.csv
```
Otras opciones de clasificación: `--model linear`, `--model rf`, `--model gb`.

#### Regresión (Puntual y por Cuantiles)
Entrena un regresor Random Forest con transformación log1p encapsulada en el target:
```bash
uv run python -m src.cli train --model rf --task regression --data data/users_train.csv
```

Entrena un modelo probabilístico por cuantiles (predice los cuantiles 10%, 50% y 90% para obtener intervalos):
```bash
uv run python -m src.cli train --model quantile --task regression --data data/users_train.csv
```

**Parámetros aceptados:**
- `--model`: `linear`, `rf`, `gb`, `xgb` o `quantile`.
- `--task`: `classification` (defecto) o `regression`.
- `--data`: Ruta de datos (CSV local o archivo `.sql` de consulta).
- `--test-size`: Tamaño del split de validación (defecto: `0.1`).
- `--use-most-common`: Si se especifica, activa la reducción de cardinalidad categórica.
- `--hyperparameters`: String JSON con hiperparámetros personalizados, por ejemplo: `'{"n_estimators": 150, "max_depth": 5}'`.

---

### 2. Comparación de Modelos (`compare`)

Entrena múltiples familias de modelos sobre el mismo conjunto de datos, genera gráficos de diagnóstico de forma individual, y produce un reporte unificado `artifacts/comparison.json` para facilitar la selección.

```bash
uv run python -m src.cli compare --models linear rf xgb --task classification --data data/users_train.csv
```

**Parámetros aceptados:**
- `--models`: Lista separada por espacios de modelos a comparar (defecto: `linear rf xgb`).
- `--task`: `classification` o `regression`.
- `--data`: Ruta del conjunto de datos.
- `--test-size`: Tamaño del split de validación.

---

### 3. Ajuste de Hiperparámetros (`hpo`)

Busca la combinación óptima de hiperparámetros utilizando Optuna mediante validación cruzada. Nota: **el conjunto de testeo está completamente aislado** y el ajuste se realiza exclusivamente sobre pliegues de entrenamiento.

```bash
uv run python -m src.cli hpo --model rf --n-trials 10 --timeout 600 --data data/users_train.csv
```

**Parámetros aceptados:**
- `--model`: Algoritmo a optimizar (`xgb`, `rf` o `linear`).
- `--task`: `classification` o `regression`.
- `--data`: Ruta de datos de entrenamiento.
- `--n-trials`: Cantidad máxima de iteraciones (defecto: de `settings.py`).
- `--timeout`: Tiempo de ejecución límite en segundos.

---

### 4. Inferencia en Producción (`predict`)

Carga el pipeline completo de inferencia serializado y genera predicciones de negocio directamente formateadas de extremo a extremo sin postprocesamientos manuales externos.

#### Usando el último modelo entrenado (latest)
```bash
uv run python -m src.cli predict --data data/users_test.csv --output data/predictions.csv --model-type xgb
```

#### Usando una corrida de entrenamiento específica (run_id)
```bash
uv run python -m src.cli predict --data data/users_test.csv --output data/predictions.csv --run-id 202607010336_xgb
```

**Parámetros aceptados:**
- `--data`: Ruta de datos crudos a predecir.
- `--output`: Ruta del archivo CSV donde se guardará el resultado estructurado de negocio.
- `--model-type`: Algoritmo del modelo si se desea cargar el más reciente (`linear`, `rf`, `gb`, `xgb`, `quantile`).
- `--run-id`: ID único de corrida si se desea un modelo histórico específico.

---

## Ejecución con Docker

El proyecto incluye un `Dockerfile` optimizado y de etapa única para facilitar su despliegue y empaquetamiento en producción.

### 1. Construir la Imagen Docker
```bash
docker build -t ml-churn-pipeline:latest .
```

### 2. Ejecutar Entrenamiento desde Docker
Montando la carpeta local `data/` y `artifacts/` para persistir los resultados locales:

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/artifacts:/app/artifacts \
  ml-churn-pipeline:latest train --model xgb --task classification --data data/users_train.csv
```

### 3. Ejecutar Inferencia de Producción desde Docker
```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/artifacts:/app/artifacts \
  -v $(pwd)/models:/app/models \
  ml-churn-pipeline:latest predict --data data/users_test.csv --output data/predictions.csv --model-type xgb
```

---

## Ejecución de Pruebas Unitarias

El entorno cuenta con `pytest` configurado para validar la lógica del pipeline.

Ejecutar las pruebas unitarias locales:
```bash
uv run pytest tests/
```
