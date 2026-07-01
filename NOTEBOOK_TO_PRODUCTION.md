# De Notebook a Código de Producción — Definiciones Genéricas

## Objetivo

Transformar un notebook monolítico (o un flujo repartido en varios notebooks) en un paquete Python profesional, modular y ejecutable por CLI.

## Alcance

- El notebook puede tener EDA, limpieza, features, entrenamiento y evaluación juntos.
- El objetivo es extraer cada etapa a módulos reutilizables dentro de `src/`.
- El resultado final debe permitir entrenamiento, evaluación, registro de artefactos, comparación de modelos y (si aplica) HPO, sin lógica crítica dentro del notebook.

## Definiciones y reglas

### 1. Capa de datos desacoplada

- La fuente inicial puede ser CSV (u otra simple), pero la lógica de lectura debe quedar aislada en `src/data/source.py` para poder reemplazar el backend después (DB, API, data warehouse) sin romper el pipeline.

### 2. Convenciones de nombres

- Variables descriptivas, sin abreviaturas opacas.
- Identificadores de código en inglés (variables, funciones, clases, constantes, claves de salida).
- Docstrings y comentarios en español neutro.

### 3. Config centralizada

- Todos los parámetros globales deben vivir en `src/config/settings.py`.
- No hardcodear umbrales, rangos, listas, tamaños de split ni hiperparámetros en el resto del código.
- Usar nombres explícitos para defaults, por ejemplo `DEFAULT_*_PARAMS`.

### 4. Modularidad estricta por etapas

- El orquestador en `src/train/pipeline.py` solo coordina etapas.
- No debe contener lógica de negocio (ni split, ni fit, ni armado de payloads).
- Separar como mínimo:
  - Lectura de datos: `src/data/source.py`
  - Sanitización/procesamiento: `src/data/processing.py`
  - Feature engineering: `src/features/build.py`
  - Entrenamiento: `src/train/training.py`
  - Métricas/evaluación: `src/train/metrics.py` y `src/train/evaluation.py`
  - Registro de artefactos: `src/train/registry.py`

### 5. Features como pipeline de sklearn

- Evitar feature engineering disperso en funciones sueltas de pandas.
- Usar un pipeline explícito (ej. `ColumnTransformer` + transformadores propios), ajustado solo con train y reutilizado idéntico en test/inferencia.

### 6. Modelos empaquetados para inferencia

- El modelo final debe serializarse junto con el pipeline de features en un único artefacto de inferencia.
- `predict()` del artefacto debe devolver salida de negocio directamente (sin postprocesamiento manual fuera del modelo).

### 7. Soporte para familias de modelos

- Incluir al menos:
  - **Modelos puntuales** (point): una predicción por registro.
  - **Modelo probabilístico** (si aplica al caso): múltiples cuantiles/intervalos.
- El CLI de entrenamiento debe aceptar `--model` para seleccionar variante.

### 8. Transformación de target encapsulada

- Si se usa `log1p`/`expm1` u otra transformación, debe quedar encapsulada dentro del estimador (ej. con `TransformedTargetRegressor`), no en scripts sueltos.

### 9. Métricas y comparación

- Métricas puntuales y probabilísticas separadas y puras (funciones en `metrics.py`).
- Mantener un comando de comparación entre familias de modelos y registrar salida estructurada (ej. `comparison.json`).

### 10. HPO desacoplado

- Implementar ajuste de hiperparámetros en `src/train/hpo.py` con su CLI.
- El estudio no debe tocar test final; validar internamente dentro de train.
- Espacio de búsqueda y `n_trials` deben venir de `settings.py`.

### 11. Artefactos por corrida

- Cada ejecución escribe en `artifacts/<RUN_ID>/` con timestamp `YYYYMMDDHHMM`.
- El formato de `RUN_ID` se define en config (`RUN_ID_FORMAT`).
- Guardar modelo, métricas y metadatos de forma consistente.

### 12. Calidad de código

- Incluir `ruff` en dependencias de desarrollo.
- Exigir `ruff check` y `ruff format` limpios sobre `src/`.

### 13. Empaquetado y ejecución

- El proyecto debe correr por CLI (`python -m ...`), sin depender del notebook.
- Mantener `Dockerfile` simple (single-stage) y evitar complejidad innecesaria.

### 14. Documentación operativa

- Documentar en `HOWTO.md` los comandos de entrenamiento, comparación, HPO, inferencia y ejecución con Docker.

### 15. Tests

- Si la etapa lo permite, dejar estructura preparada para tests.
- Si no es parte del alcance actual, explicitar que se incorporan en fase siguiente.

## Entregable esperado

- Código modular en `src/` listo para producción.
- CLI funcional para `train` / `compare` / `hpo`.
- Artefactos versionados por corrida.
- Notebook opcional solo como apoyo exploratorio, no como pieza operativa crítica.
