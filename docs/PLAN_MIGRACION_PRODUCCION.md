# Plan de Migración a Código Productivo — Churn Model (Flood-It)

**Fecha:** 2026-07-01
**Autor:** creyesp
**Alcance acordado:** Refactor de notebooks → paquete Python instalable, con **batch scoring** por CLI, tests y ejecución **local** para validar. Se migra **solo el código del modelo ganador (XGBoost)** y se congela; el resto de modelos (linear/rf/quantile) y las mejoras del spec quedan fuera de alcance. Sin API online, sin orquestación cloud, sin monitoreo/retraining.

### Decisiones cerradas
1. **Modelo a migrar:** solo el **ganador → XGBoost** (`xgb_model_full`: preprocessor + `XGBClassifier` con `scale_pos_weight` + params de Optuna). Se descartan `latest_linear`, `latest_rf`, `latest_quantile`.
2. **Salida del scoring:** archivo **local** (CSV) — sin BigQuery en el path.
3. **Umbral:** default **0.5**, expuesto como **parámetro configurable** (flag CLI / arg de función) para ajustar después.

---

## 1. Situación actual (lo que hay)

- **Lógica en notebooks**: `notebooks/0.0_data_collection` → `5.0_explanability`. Todo el flujo (colección, EDA, procesamiento, modelado linear/bagging/boosting/xgb, tuning, explicabilidad) vive en notebooks, no reutilizable ni testeable.
- **Código fuente mínimo**: `notebooks/src/` con `config.py` (columnas), `transformer.py` (preprocessor sklearn + `MostCommonCategories`), `utils.py` (plots de métricas). Import relativo, atado a notebooks.
- **Datos**: generados en BigQuery vía `data/queries/flood_it_dataset.sql` → CSVs `users_train.csv` (7190), `users_test.csv` (799), `users_raw.csv`, `users_not_yet.csv`.
- **Features**: 11 numéricas `cnt_*` + 3 categóricas (`country_name`, `device_os`, `device_lang`). Label `churned` (no vuelve en 24h). Columnas a ignorar: `user_first_engagement`, `user_pseudo_id`, `is_enable`, `bounced`.
- **Modelos**: `.joblib` en `models/` (`latest_linear`, `latest_rf`, `latest_xgb`, `latest_quantile`). El **ganador es XGBoost** (`xgb_model_full`, notebooks `4.0_hp_xgboost` + `5.0_explanability`). Sin metadata ni versionado formal. **Solo se migra el XGBoost.**
- **Serving**: `serving/` es un esqueleto MLServer (echo, no funcional). **Fuera de alcance.**
- **Tooling ya presente**: `pyproject.toml` (uv), `Makefile` (lint/format/type-check con ruff+mypy), devcontainer, `.venv`.

### Problemas para producción
1. Lógica no importable ni testeable (vive en celdas).
2. Sin punto de entrada reproducible para entrenar ni para puntuar.
3. Preprocessor con bugs menores y typos (`categical`, `Redece cadinality`, `print` de debug en `fit`).
4. `utils.py` usa APIs de sklearn removidas (`metrics.plot_confusion_matrix`, `plot_roc_curve`) → roto en sklearn ≥1.2.
5. Modelos guardados sin metadata (versión sklearn, hash de datos, métricas, features).
6. Rutas y config hardcodeadas en notebooks.

---

## 2. Objetivo

Convertir el proyecto en un **paquete Python instalable** (`floodit_churn`) con:
- Código de datos, features, entrenamiento y **batch scoring** separado y testeado.
- Dos CLIs: `train` (reentrena y congela artefacto) y `score` (batch scoring de un CSV → CSV de predicciones).
- Ejecución **100% local** reproducible (`uv run ...`), sin dependencias cloud en el path crítico.
- El **modelo actual congelado** como artefacto versionado con metadata.

**No incluye:** API REST/MLServer, orquestación (Airflow/cron cloud), monitoreo/drift, retraining automático, mejoras de modelado del spec.

---

## 3. Arquitectura destino (estructura de paquete)

```
src/floodit_churn/
├── __init__.py
├── config.py            # columnas, rutas, constantes (migrado de notebooks/src/config.py)
├── data.py              # carga/validación de CSV, split, checks de esquema
├── features.py          # preprocessor sklearn + MostCommonCategories (limpiado de transformer.py)
├── model.py             # build_pipeline() con XGBClassifier (params ganadores), train(), evaluate(), save/load con metadata
├── scoring.py           # batch scoring: CSV entrada -> DataFrame con proba + label (umbral configurable, default 0.5)
├── metrics.py           # métricas y (opcional) plots, sin APIs removidas de sklearn
└── cli.py               # entrypoints: floodit-train / floodit-score

tests/
├── test_data.py         # esquema, columnas, tipos
├── test_features.py     # preprocessor fit/transform, MostCommonCategories
├── test_model.py        # pipeline entrena y predice; round-trip save/load
└── test_scoring.py      # score de un CSV chico produce columnas esperadas

artifacts/
└── model_v1/            # modelo congelado + metadata.json (features, sklearn ver, data hash, métricas)

data/                    # sin cambios (CSVs existentes)
docs/                    # este plan + spec existente
```

Reemplaza a `notebooks/src/` (los notebooks pasan a importar del paquete, no al revés).

---

## 4. Fases y tareas

### Fase 0 — Preparación (0.5 día)
- [ ] Crear rama `feat/productionize`.
- [ ] Añadir layout `src/floodit_churn/` y `[project.scripts]` en `pyproject.toml` (`floodit-train`, `floodit-score`).
- [ ] Configurar `pytest` en `pyproject.toml` y target `test` en el `Makefile`.
- [ ] Pin de datos: registrar hash SHA256 de `users_train.csv` / `users_test.csv` en `artifacts/`.

### Fase 1 — Migrar código base desde notebooks/src (1 día)
- [ ] `config.py`: mover columnas + añadir rutas (`DATA_DIR`, `ARTIFACTS_DIR`) y semilla global.
- [ ] `features.py`: portar `preprocessor()` y `MostCommonCategories`; corregir typos (`categorical`), quitar `print` de debug, arreglar `transform` para no mutar el input in-place, añadir docstrings/tipos.
- [ ] `data.py`: `load_dataset(path)` con validación de esquema (columnas esperadas, tipos, sin nulos en label), `split()` estratificado reutilizando el split existente como test congelado.
- [ ] `metrics.py`: reemplazar `plot_confusion_matrix`/`plot_roc_curve` (removidos) por `ConfusionMatrixDisplay`/`RocCurveDisplay`; mantener `precision_recall_vs_thr`.

### Fase 2 — Entrenamiento reproducible + congelar XGBoost (1 día)
- [ ] `model.py`: `build_pipeline()` = preprocessor + `XGBClassifier(**best_params, scale_pos_weight=..., random_state=42)` con los params ganadores extraídos de `4.0_hp_xgboost`; `train(df)`, `evaluate(model, df)`.
- [ ] `save_model(model, path, metadata)` / `load_model(path)`: persistir con `metadata.json` (features, versión sklearn/xgboost, hash de datos, métricas, fecha, git SHA, umbral default).
- [ ] Reproducir el **XGBoost ganador** (`xgb_model_full`) con semilla fija y guardarlo en `artifacts/model_v1/` como artefacto congelado + su `metadata.json`.
- [ ] Verificar que las métricas del artefacto reproducido coinciden (dentro de tolerancia) con las del notebook `4.0`/`5.0`.

### Fase 3 — Batch scoring (0.5 día)
- [ ] `scoring.py`: `score_csv(input_csv, model_path, threshold=0.5) -> DataFrame` con `user_pseudo_id`, `churn_proba`, `churn_pred`. Umbral **default 0.5**, parametrizable.
- [ ] CLI `floodit-score --input data/users_test.csv --model artifacts/model_v1 --output preds.csv --threshold 0.5` (escribe a archivo local; `--threshold` opcional para ajustar después).
- [ ] CLI `floodit-train --data data/users_train.csv --out artifacts/model_v2` (para reentrenos manuales del XGBoost).

### Fase 4 — Tests + CI local (1 día)
- [ ] Tests unitarios de las 4 áreas (ver estructura). Incluir un fixture con un CSV mínimo (~20 filas).
- [ ] Test de regresión: cargar `artifacts/model_v1` y verificar que puntúa un CSV conocido con proba esperada (evita romper el modelo congelado en refactors).
- [ ] `make lint`, `make format --check`, `make type-check`, `make test` en verde.
- [ ] (Opcional) GitHub Actions que corra lint + mypy + pytest en cada push.

### Fase 5 — Documentación y limpieza (0.5 día)
- [ ] `README.md`: sección "Uso productivo local" (instalar, entrenar, puntuar).
- [ ] Reapuntar `notebooks/src/*` para importar de `floodit_churn` (o dejar los notebooks como referencia/EDA y deprecar `notebooks/src`).
- [ ] Actualizar `.gitignore` para artefactos/outputs de scoring.

**Estimación total: ~4.5–5 días.**

---

## 5. Criterios de aceptación (Definition of Done)

1. `uv run floodit-score --input data/users_test.csv --model artifacts/model_v1 --output preds.csv` corre en local y genera un CSV con `user_pseudo_id, churn_proba, churn_pred`.
2. `uv run floodit-train ...` reproduce un artefacto con `metadata.json` completo.
3. El artefacto `model_v1` reproduce las métricas del modelo actual dentro de tolerancia (test de regresión en verde).
4. `make lint`, `make type-check`, `make test` pasan sin errores.
5. Cero lógica de negocio nueva en notebooks: todo importable desde `floodit_churn`.
6. El modelo actual queda **congelado y versionado** (no se cambia el algoritmo ni las features).

---

## 6. Riesgos y decisiones

| Riesgo | Mitigación |
|---|---|
| El `.joblib` actual no reproduce por versión de sklearn distinta | Reentrenar con semilla fija y validar métricas; guardar versión en metadata. |
| `MostCommonCategories.transform` muta el array in-place (efectos raros) | Copiar el array antes de transformar en el refactor. |
| `utils.py` roto por APIs removidas de sklearn | Migrar a las clases `*Display` en `metrics.py`. |
| Rutas relativas de notebooks vs paquete | Centralizar rutas en `config.py` basadas en raíz del repo. |
| Params ganadores de Optuna dispersos en el notebook (`study.trials[1].params`) | Extraerlos explícitamente y fijarlos como constante en `model.py` (no re-tunear). |

---

## 7. Preguntas abiertas

Ninguna bloqueante — las tres decisiones clave están cerradas (ver "Decisiones cerradas" en el encabezado):
- Modelo ganador = **XGBoost** (único a migrar).
- Salida = **CSV local**.
- Umbral = **0.5 parametrizable**.

Pendiente menor: confirmar los **hiperparámetros exactos** del XGBoost ganador a fijar (se extraen de `notebooks/4.0_hp_xgboost.ipynb`; hay ambigüedad entre `study.best_trial.params` y `study.trials[1].params` en el notebook — resolver al migrar).

---

## 8. Fuera de alcance (fases futuras)

- API online / MLServer real (`serving/`).
- Orquestación / scheduling (cron, Airflow, Vertex Pipelines).
- Monitoreo, drift y retraining automático.
- Mejoras de modelado del `docs/datapowers/specs/2026-06-04-churn-model-improvement-spec.md` (PR-AUC, leakage audit, threshold por costo, calibración).
- Ingesta directa desde BigQuery en el path productivo.
