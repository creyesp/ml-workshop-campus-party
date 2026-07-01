# Flood-It Churn Model — Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `datapowers:subagent-driven-experimentation` (recommended) or `datapowers:executing-experiment-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **REQUIRED FOR EVERY TASK:** `datapowers:test-driven-modeling` (test before code), `datapowers:experiment-tracking` (log SHA + data hash + params + metrics).

**Goal:** Harden the existing Flood-It churn pipeline so "the model improved" is a defensible claim — PR-AUC with bootstrap CIs, a documented baseline to beat, a leakage audit, and a reproducible, tracked workflow.

**Spec:** `docs/datapowers/specs/2026-06-04-churn-model-improvement-spec.md`

**Architecture:** A new installable package `src/floodit/` holds pure, testable units (data load + contract, preprocessing factory, validation splits, baselines, tuned XGBoost, evaluation toolkit, leakage audit, tracking wrapper). The legacy notebooks/SQL stay frozen as reference. Experiments are driven by thin scripts under `experiments/` that compose these units and log to a **local MLflow file store** (`mlruns/`). Headline numbers come from k-fold CV on `users_train.csv`; the frozen `users_test.csv` is scored once per candidate.

**Tech Stack:** Managed with **uv** in a project-local `.venv`. Python 3.12; modern, wheel-available pins: pandas ≥2.0, numpy ≥1.26, scikit-learn ≥1.4, xgboost ≥2.0, optuna ≥3.5, mlflow ≥2.10, pytest ≥8, matplotlib ≥3.8. **Decision (2026-06-04):** the workshop's original 3.9 + sklearn 0.24.2 stack is *not* used — this machine has no Python 3.9 and those old versions lack wheels here. The legacy notebooks/`notebooks/src` stay frozen as reference; the new `floodit` package is written against the modern stack. All commands run via `.venv/bin/python` (or `uv run`).

> **GLOBAL CONSTRAINT — parallelism.** Never use `n_jobs=-1`. A single constant `floodit.config.N_JOBS` (default `2`, overridable via `FLOODIT_N_JOBS` env var) is threaded through **every** `cross_val_*`, `GridSearchCV/cross_validate`, `study.optimize(n_jobs=...)`, and estimator `n_jobs=`. This leaves CPU for other processes on the machine. Any task that fits a model or runs CV MUST pass `n_jobs=N_JOBS`. Grep check before each commit: `grep -rn "n_jobs" src/ experiments/ | grep -- "-1"` must return nothing.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, pytest config, pin new deps (mlflow, pytest) |
| `src/floodit/__init__.py` | Package marker |
| `src/floodit/config.py` | Column lists (ported from `notebooks/src/config.py`), `RANDOM_SEED`, `N_JOBS`, paths |
| `src/floodit/data/load.py` | Load a pinned CSV, return DataFrame + verify sha256 hash |
| `src/floodit/data/contract.py` | Schema/type/range/nullability/balance assertions on the loaded frame |
| `src/floodit/features/preprocess.py` | Pure `build_preprocessor()` ColumnTransformer factory (ported + cleaned from `notebooks/src/transformer.py`) |
| `src/floodit/models/split.py` | `stratified_kfold()` and `time_based_split()` — validation strategy, committed before any fit |
| `src/floodit/models/baseline.py` | `majority_baseline()` (trivial) + `logreg_baseline()` (strong, class-balanced) |
| `src/floodit/models/xgb.py` | `build_xgb()` + Optuna `tune_xgb()` (CV, `n_jobs=N_JOBS`) |
| `src/floodit/evaluate/metrics.py` | PR-AUC, ROC-AUC, Brier, `bootstrap_ci()`, `recall_at_precision()`, `threshold_at_precision()` |
| `src/floodit/evaluate/segments.py` | Slice metrics by `country_name` / `device_os` |
| `src/floodit/evaluate/leakage.py` | Ablation + permutation-importance audit (H3 tripwire) |
| `src/floodit/tracking.py` | MLflow wrapper: log code SHA, data hash, params, metrics, artifacts |
| `experiments/00_pin_data.py` | Compute + record dataset hashes |
| `experiments/01_leakage_audit.py` | Run H3 audit, write report |
| `experiments/02_baselines.py` | CV-score trivial + strong baseline, log to MLflow |
| `experiments/03_threshold_tuning.py` | H1: cost/precision threshold vs 0.5 |
| `experiments/04_xgb_tuned.py` | H2: Optuna XGBoost, CV CIs vs baseline |
| `experiments/05_final_test.py` | Score winning candidate(s) once on frozen test + segment metrics |
| `experiments/06_repro_check.py` | Re-run end-to-end from clean env, match metrics |
| `tests/...` | One test file per source file |
| `docs/datapowers/reports/2026-06-04-churn-results.md` | Final decision report (Section 10 criteria) |

One responsibility per file. Files that change together live together.

---

## Phase 0 — Project scaffold

### Task 0: Installable package + tooling

**Spec section:** §6 (reproducibility), global parallelism constraint

**Files:**
- Create: `pyproject.toml`, `src/floodit/__init__.py`, `src/floodit/config.py`
- Create: `tests/__init__.py`, `tests/test_config.py`

- [ ] **Step 1: Write the failing test** — `tests/test_config.py`
```python
from floodit import config

def test_n_jobs_is_capped_not_all_cores():
    assert config.N_JOBS != -1
    assert 1 <= config.N_JOBS <= 4

def test_seed_and_columns_present():
    assert config.RANDOM_SEED == 42
    assert config.LABEL_COLUMN == "churned"
    assert len(config.NUMERICAL_COLUMNS) == 11
    assert config.CATEGORICAL_COLUMNS == ["country_name", "device_os", "device_lang"]
```

- [ ] **Step 2: Run to verify it fails**
Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: floodit`.

- [ ] **Step 3: Implement minimal code**
Create `pyproject.toml` (setuptools, package dir `src`, `[tool.pytest.ini_options] pythonpath=["src"]`, deps add `mlflow`, `pytest`). Create `src/floodit/config.py` porting column lists from `notebooks/src/config.py`, plus:
```python
import os
RANDOM_SEED = 42
N_JOBS = int(os.environ.get("FLOODIT_N_JOBS", "2"))  # never -1; leave CPU for other processes
```
Add `DATA_DIR`, `MLRUNS_DIR` path constants.

- [ ] **Step 4: Run to verify it passes**
Run: `pip install -e . && pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add pyproject.toml src/floodit tests/test_config.py
git commit -m "task 0: floodit package scaffold + capped N_JOBS config"
```

---

## Phase 1 — Data layer

### Task 1: Hash-pinned data loader

**Spec section:** §6 (version pin)

**Files:**
- Create: `src/floodit/data/__init__.py`, `src/floodit/data/load.py`, `tests/data/test_load.py`

- [ ] **Step 1: Write the failing test**
```python
from floodit.data.load import sha256_of, load_dataset

def test_sha256_is_deterministic(tmp_path):
    p = tmp_path / "x.csv"; p.write_text("a,b\n1,2\n")
    assert sha256_of(p) == sha256_of(p)

def test_load_train_shape_and_balance():
    df = load_dataset("train", verify=False)
    assert df.shape == (7190, 19)
    assert round(df["churned"].mean(), 3) == 0.231
```

- [ ] **Step 2: Run to verify it fails** — `pytest tests/data/test_load.py -v` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement minimal code** — `load.py`:
  - `sha256_of(path) -> str` (stream file in chunks).
  - `load_dataset(name: str, verify: bool=True) -> DataFrame`: map `"train"/"test"/"raw"` to `data/users_*.csv`; if `verify`, assert hash equals the value pinned in `config.DATA_HASHES` (filled by Task 2); parse `user_first_engagement` as datetime.

- [ ] **Step 4: Run to verify it passes** — `pytest tests/data/test_load.py -v` → PASS.

- [ ] **Step 5: Commit** — `git commit -m "task 1: hash-pinned dataset loader"`

### Task 2: Pin the dataset hashes

**Spec section:** §6 (version pin, record hashes)

**Files:**
- Create: `experiments/00_pin_data.py`
- Modify: `src/floodit/config.py` (add `DATA_HASHES = {...}`)

- [ ] **Step 1: Compute hashes** — `experiments/00_pin_data.py` prints `sha256_of` for `users_train.csv`, `users_test.csv`, `users_raw.csv`.
Run: `python experiments/00_pin_data.py`
Expected output to inspect: three 64-char hex digests.

- [ ] **Step 2: Record** — paste the three digests into `config.DATA_HASHES`.

- [ ] **Step 3: Verify pin enforced** — add `tests/data/test_load.py::test_verify_true_passes` calling `load_dataset("train", verify=True)`.
Run: `pytest tests/data/test_load.py -v` → PASS.

- [ ] **Step 4: Commit** — `git commit -m "task 2: pin dataset sha256 hashes"`

### Task 3: Data contract

**Spec section:** §2 (edge cases), §6 (balance), and `datapowers:data-contract-validation`

**Files:**
- Create: `src/floodit/data/contract.py`, `tests/data/test_contract.py`

- [ ] **Step 1: Write the failing test**
```python
import pandas as pd, pytest
from floodit.data.contract import validate_contract
from floodit.data.load import load_dataset

def test_real_train_passes_contract():
    validate_contract(load_dataset("train", verify=False))  # no raise

def test_negative_count_rejected():
    df = load_dataset("train", verify=False).copy()
    df.loc[0, "cnt_user_engagement"] = -1
    with pytest.raises(AssertionError):
        validate_contract(df)
```

- [ ] **Step 2: Run to verify it fails** → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement** — `validate_contract(df)` asserts: all `cnt_*` integer ≥ 0; `churned ∈ {0,1}`; `is_enable` all 1 and `bounced` all 0 (spec §2 exclusions hold); no nulls in `cnt_*`; categorical columns are strings (nulls allowed → imputed downstream); churn prevalence within `[0.20, 0.26]`.

- [ ] **Step 4: Run to verify it passes** → PASS.

- [ ] **Step 5: Commit** — `git commit -m "task 3: data contract validation"`

---

## Phase 2 — Validation strategy (precedes any model fit)

### Task 4: Splits — stratified k-fold + time-based

**Spec section:** §6 (split strategy), `datapowers:validation-strategy-design`

**Files:**
- Create: `src/floodit/models/__init__.py`, `src/floodit/models/split.py`, `tests/models/test_split.py`

- [ ] **Step 1: Write the failing test**
```python
import numpy as np
from floodit.models.split import stratified_kfold, time_based_split
from floodit.data.load import load_dataset
from floodit import config

def test_stratified_kfold_preserves_prevalence():
    df = load_dataset("train", verify=False)
    y = df[config.LABEL_COLUMN].values
    for tr, va in stratified_kfold(df, y, n_splits=5):
        assert abs(y[va].mean() - y.mean()) < 0.03
        assert set(tr).isdisjoint(set(va))

def test_time_split_is_ordered_by_first_engagement():
    df = load_dataset("train", verify=False)
    tr, va = time_based_split(df, frac=0.8)
    assert df.iloc[tr]["user_first_engagement"].max() <= df.iloc[va]["user_first_engagement"].min()
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement** — `stratified_kfold` wraps `StratifiedKFold(n_splits, shuffle=True, random_state=RANDOM_SEED)`; `time_based_split` sorts by `user_first_engagement` and cuts at `frac`. Document in module docstring why per-user rows justify k-fold and why a time split is the robustness check (spec §6).

- [ ] **Step 4: Run to verify it passes** → PASS.

- [ ] **Step 5: Commit** — `git commit -m "task 4: validation splits (stratified kfold + time-based)"`

---

## Phase 3 — Preprocessing

### Task 5: Pure preprocessor factory

**Spec section:** §3 (features at prediction time), `datapowers:feature-engineering-systematically`

**Files:**
- Create: `src/floodit/features/__init__.py`, `src/floodit/features/preprocess.py`, `tests/features/test_preprocess.py`

- [ ] **Step 1: Write the failing test**
```python
from floodit.features.preprocess import build_preprocessor
from floodit.data.load import load_dataset
from floodit import config

def test_preprocessor_transforms_without_leaking_ignore_cols():
    df = load_dataset("train", verify=False)
    pre = build_preprocessor()
    X = pre.fit_transform(df)
    assert X.shape[0] == len(df)
    # ignored identity/time columns must not appear as features
    names = pre.get_feature_names_out() if hasattr(pre, "get_feature_names_out") else []
    assert all("user_pseudo_id" not in str(n) for n in names)
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement** — port `notebooks/src/transformer.py::preprocessor` into `build_preprocessor()`: numeric impute(0)+scale, categorical impute(most_frequent)+optional MostCommonCategories+OneHot(handle_unknown="ignore"), drop `IGNORE_COLUMNS`. Keep it a pure factory (no global state).

- [ ] **Step 4: Run to verify it passes** → PASS.

- [ ] **Step 5: Commit** — `git commit -m "task 5: pure preprocessor factory"`

---

## Phase 4 — Evaluation toolkit (precedes baseline scoring)

### Task 6: Metrics + bootstrap CIs + threshold@precision

**Spec section:** §4 (metrics), §5 (CI rule), `datapowers:model-evaluation-rigorously`

**Files:**
- Create: `src/floodit/evaluate/__init__.py`, `src/floodit/evaluate/metrics.py`, `tests/evaluate/test_metrics.py`

- [ ] **Step 1: Write the failing test** (hand-computed expectations)
```python
import numpy as np
from floodit.evaluate.metrics import pr_auc, bootstrap_ci, threshold_at_precision, recall_at_precision

def test_pr_auc_perfect_separation_is_one():
    y = np.array([0,0,1,1]); p = np.array([0.1,0.2,0.8,0.9])
    assert pr_auc(y, p) == 1.0

def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0,2,500); p = rng.random(500)
    lo, mid, hi = bootstrap_ci(y, p, metric=pr_auc, n=200, seed=0)
    assert lo <= mid <= hi

def test_threshold_at_precision_monotone():
    y = np.array([0,0,0,1,1]); p = np.array([0.1,0.3,0.4,0.6,0.9])
    thr = threshold_at_precision(y, p, target=0.6)
    assert 0.0 <= thr <= 1.0
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement** — `pr_auc=average_precision_score`, `roc_auc`, `brier=brier_score_loss`; `bootstrap_ci(y,p,metric,n=1000,seed)` resampling indices → returns (2.5th, 50th, 97.5th pct); `threshold_at_precision(y,p,target)` from the PR curve; `recall_at_precision(y,p,target)`. Use `RANDOM_SEED` default; no `n_jobs` here.

- [ ] **Step 4: Run to verify it passes** → PASS.

- [ ] **Step 5: Commit** — `git commit -m "task 6: evaluation metrics + bootstrap CIs"`

### Task 7: Segment metrics

**Spec section:** §8 (fairness flag — segment reporting), `datapowers:fairness-and-bias-audit` (report only)

**Files:**
- Create: `src/floodit/evaluate/segments.py`, `tests/evaluate/test_segments.py`

- [ ] **Step 1: Write the failing test** — `segment_metrics(df, y_true, y_prob, by="device_os")` returns a DataFrame with one row per group and columns `n, prevalence, pr_auc`.
- [ ] **Step 2: Run to verify it fails** → FAIL.
- [ ] **Step 3: Implement** `segment_metrics`.
- [ ] **Step 4: Run to verify it passes** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "task 7: segment-level metrics"`

### Task 8: MLflow tracking wrapper

**Spec section:** §8 (reproducibility), `datapowers:experiment-tracking`

**Files:**
- Create: `src/floodit/tracking.py`, `tests/test_tracking.py`

- [ ] **Step 1: Write the failing test** — `start_run(name)` context manager logs `git_sha`, `data_hash`, params, metrics into a temp `mlruns/`; test asserts an MLflow run is created with those tags. Set tracking URI to `tmp_path`.
- [ ] **Step 2: Run to verify it fails** → FAIL.
- [ ] **Step 3: Implement** — wrapper around `mlflow.start_run`; auto-tag `git rev-parse HEAD` (via subprocess) and `config.DATA_HASHES`. Local file store at `config.MLRUNS_DIR`.
- [ ] **Step 4: Run to verify it passes** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "task 8: mlflow tracking wrapper (sha + data hash)"`

---

## Phase 5 — Leakage audit (H3 tripwire, BEFORE trusting any score)

### Task 9: Leakage audit on `cnt_*` features

**Spec section:** §7 H3, §8 (window leakage), `datapowers:preventing-data-leakage`

**Files:**
- Create: `src/floodit/evaluate/leakage.py`, `tests/evaluate/test_leakage.py`
- Create: `experiments/01_leakage_audit.py`
- Create: `docs/datapowers/reports/2026-06-04-leakage-audit.md`

> **Context for the worker:** In `data/queries/flood_it_dataset.sql`, `churned` is derived from `user_last_engagement` (global MAX over all `user_engagement` events), while `cnt_user_engagement` counts `user_engagement` events with `timestamp <= first+24h`. A retained user (churned=0) is defined as still engaging at/after 24h. So `cnt_user_engagement` (and other `cnt_*` that ride on engagement near the boundary) may partly encode the label. This task quantifies that — it is the experiment's tripwire.

- [ ] **Step 1: Write the failing test**
```python
from floodit.evaluate.leakage import single_feature_auc, ablation_drop
# single_feature_auc(df, feature, y) returns ROC-AUC of that feature alone
def test_single_feature_auc_runs():
    ...  # assert returns float in [0,1] for cnt_user_engagement
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement** — `single_feature_auc` (univariate ROC-AUC per feature), `ablation_drop(model_factory, df, y, feature)` (CV PR-AUC with vs without a feature, using `stratified_kfold` + `N_JOBS`), and permutation importance on the LogReg baseline.

- [ ] **Step 4: Run to verify it passes** → PASS.

- [ ] **Step 5: Run the audit experiment**
Run: `python experiments/01_leakage_audit.py`
Expected output to inspect: per-feature univariate AUC table + ablation deltas. **Decision rule (tripwire):** if any single `cnt_*` reaches univariate ROC-AUC ≥ 0.85 **or** removing one feature collapses CV PR-AUC toward prevalence (≈0.23), flag it in the report and **halt modeling** pending feature redefinition (spec §4 tripwire, §10 Kill). Otherwise document "no disqualifying leakage" and proceed.

- [ ] **Step 6: Write `docs/datapowers/reports/2026-06-04-leakage-audit.md`** with the table, the decision, and which features (if any) are quarantined.

- [ ] **Step 7: Commit** — `git commit -m "task 9: leakage audit (H3) + report"`

---

## Phase 6 — Baselines (documented number to beat)

### Task 10: Trivial + strong baseline, CV-scored

**Spec section:** §5 (baselines), `datapowers:baseline-first-modeling`

**Files:**
- Create: `src/floodit/models/baseline.py`, `tests/models/test_baseline.py`
- Create: `experiments/02_baselines.py`

- [ ] **Step 1: Write the failing test**
```python
from floodit.models.baseline import majority_baseline, logreg_baseline
def test_majority_predicts_negative_class():
    clf = majority_baseline()  # DummyClassifier strategy="most_frequent"
    ...  # predict_proba returns ~prevalence; test it's a valid estimator
def test_logreg_is_class_balanced():
    pipe = logreg_baseline()
    assert pipe.named_steps["clf"].class_weight == "balanced"
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement** — `majority_baseline()=DummyClassifier(strategy="most_frequent")`; `logreg_baseline()=Pipeline([build_preprocessor(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED)])`.

- [ ] **Step 4: Run to verify it passes** → PASS.

- [ ] **Step 5: Run baseline experiment**
Run: `python experiments/02_baselines.py`
Expected output to inspect: CV PR-AUC mean + bootstrap CI for both baselines, logged to MLflow. Trivial PR-AUC ≈ prevalence ≈ 0.23; strong LogReg should clear it. **Record the strong-baseline CI — this is the number to beat (spec §5).**

- [ ] **Step 6: Commit** — `git commit -m "task 10: trivial + strong baselines, CV-scored to MLflow"`

---

## Phase 7 — Threshold tuning (H1)

### Task 11: Cost/precision-based operating point

**Spec section:** §7 H1, §4 (operating point)

**Files:**
- Create: `experiments/03_threshold_tuning.py`
- Modify: `src/floodit/evaluate/metrics.py` only if a helper is missing

- [ ] **Step 1: Define the falsifiable check (in the script docstring)** — H1 holds iff cost/precision-tuned threshold improves **recall @ precision ≥ 0.60** over the 0.5 threshold, on cross-validated folds.

- [ ] **Step 2: Implement** — for the strong baseline, per CV fold: compute recall@precision≥0.60 at the tuned threshold vs recall at 0.5; aggregate mean ± bootstrap CI.

- [ ] **Step 3: Run**
Run: `python experiments/03_threshold_tuning.py`
Expected output to inspect: table `{threshold_rule, precision, recall}` for `0.5` vs `@p≥0.60`; MLflow-logged. Record H1 verdict (improve / reject).

- [ ] **Step 4: Commit** — `git commit -m "task 11: threshold tuning (H1) recall@precision"`

---

## Phase 8 — Tuned XGBoost (H2)

### Task 12: XGBoost model + Optuna tuning (capped parallelism)

**Spec section:** §7 H2, §5 (CI rule)

**Files:**
- Create: `src/floodit/models/xgb.py`, `tests/models/test_xgb.py`
- Create: `experiments/04_xgb_tuned.py`

- [ ] **Step 1: Write the failing test**
```python
from floodit.models.xgb import build_xgb
from floodit import config
def test_build_xgb_uses_capped_njobs():
    model = build_xgb({"max_depth": 3})
    assert model.named_steps["clf"].get_params()["n_jobs"] == config.N_JOBS
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement** — `build_xgb(params)` = Pipeline([build_preprocessor(), XGBClassifier(..., n_jobs=N_JOBS, random_state=RANDOM_SEED, eval_metric="logloss")]); `tune_xgb(df, y, n_trials)` = Optuna study maximizing CV PR-AUC via `stratified_kfold`, `study.optimize(..., n_jobs=N_JOBS)`. **No `n_jobs=-1` anywhere.**

- [ ] **Step 4: Run to verify it passes** → PASS.

- [ ] **Step 5: Run tuning experiment**
Run: `python experiments/04_xgb_tuned.py` (start with `n_trials=30`)
Expected output to inspect: best CV PR-AUC mean + bootstrap CI, logged with params + SHA + data hash. **H2 verdict:** XGBoost wins iff its PR-AUC CI does **not** overlap the strong-baseline CI (spec §5, §7 H2). If CIs overlap → keep the simpler model, mark H2 rejected.

- [ ] **Step 6: Commit** — `git commit -m "task 12: tuned XGBoost (H2), CV CIs vs baseline"`

---

## Phase 9 — Final held-out scoring

### Task 13: Score candidate(s) once on frozen test + segment metrics

**Spec section:** §6 (frozen test), §4 (guardrails), §8 (segment reporting)

**Files:**
- Create: `experiments/05_final_test.py`

- [ ] **Step 1: Guard against test reuse** — script asserts it runs only the agreed candidates and logs an MLflow tag `final_eval=true`; document that the test set is scored once per candidate.

- [ ] **Step 2: Implement** — refit chosen model(s) on full train, score `users_test.csv` (verify=True hash): PR-AUC + bootstrap CI, ROC-AUC, Brier, recall@precision≥0.60, confusion matrix at the tuned threshold, and `segment_metrics` by `country_name`/`device_os`.

- [ ] **Step 3: Run**
Run: `python experiments/05_final_test.py`
Expected output to inspect: headline PR-AUC ± CI on test; segment table. **Tripwire (spec §4):** if test CIs overlap across candidates (n≈799 too small to separate), report the CV comparison as primary and flag the limitation — do not claim a point-estimate winner.

- [ ] **Step 4: Commit** — `git commit -m "task 13: final frozen-test scoring + segment metrics"`

---

## Phase 10 — Reproducibility + decision report

### Task 14: Reproducibility verification

**Spec section:** §8 (reproducibility), `datapowers:reproducibility-verification`

**Files:**
- Create: `experiments/06_repro_check.py`

- [ ] **Step 1: Implement** — fresh virtualenv from `pyproject.toml`, `pip install -e .`, re-run baselines + best model, assert metrics match the logged MLflow run within tolerance (e.g. |Δ PR-AUC| < 0.005).

- [ ] **Step 2: Run**
Run: `python experiments/06_repro_check.py`
Expected output to inspect: "MATCH within tolerance" for each metric.

- [ ] **Step 3: Commit** — `git commit -m "task 14: reproducibility verification"`

### Task 15: Results & decision report

**Spec section:** §10 (decision criteria), §11 (open questions)

**Files:**
- Create: `docs/datapowers/reports/2026-06-04-churn-results.md`

- [ ] **Step 1: Write the report** — baseline CI, XGBoost CI, CI-overlap verdict, H1/H2/H3 outcomes, leakage-audit conclusion, segment metrics, and the **Deploy / Kill / Iterate** decision per spec §10. Restate open questions Q1–Q3 (cost ratio, volume budget, time-split-as-primary) for the growth team.

- [ ] **Step 2: Commit** — `git commit -m "task 15: churn improvement results + decision report"`

---

## Plan Self-Review

1. **Specs-traceability:** §1→T9/reports, §2→T3, §3→T5, §4→T6/T11/T13, §5→T10/T12, §6→T1/T2/T4/T13, §7 H1→T11 / H2→T12 / H3→T9, §8→T8/T9/T14 + T7 (fairness flag), §9 out-of-scope respected (no serving/drift tasks), §10→T15, §11→T15. ✅
2. **Granularity:** each step is one action; experiments are the only >5-min runs and are isolated as their own steps. ✅
3. **Verification:** every task has a runnable `pytest`/`python experiments/...` command + expected output. ✅
4. **Baseline precedes complex models:** Task 10 (baselines) before Task 12 (XGBoost). ✅
5. **Validation strategy precedes fitting:** Task 4 before Tasks 10/12. ✅
6. **Reproducibility precedes any deploy claim:** Task 14 before Task 15. ✅
7. **No surprises / no silent parallelism:** global `N_JOBS` constraint stated up front and re-checked per fitting task. ✅

## Transition

Hand off to `datapowers:subagent-driven-experimentation` (preferred) or `datapowers:executing-experiment-plans` to execute task-by-task.
