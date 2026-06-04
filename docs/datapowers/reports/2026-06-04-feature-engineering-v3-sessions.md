# Session-Level Features v3 — Flood-It Churn

**Date:** 2026-06-04
**Builds on:** `2026-06-04-feature-engineering-v2.md` (v2 champion: XGBoost, CV PR-AUC 0.471)
**Query:** `data/queries/flood_it_features_v3_sessions.sql` · **Data:** `data/users_features_v3_sessions.csv` (sha256 `0969ed26…d2df`)
**Experiments:** `experiments/10_features_v3.py` (sweep), `experiments/11_final_v3.py` (test)

## What changed

This 2018 Firebase export has **no `ga_session_id`** (it predates GA4 sessions —
the v2 `num_sessions` came back all-zero). Sessions are instead derived by
**time-gap sessionization**: a new session starts after a >30 min idle gap. New
features (joined onto the same split, all within the 24 h window, anchored to its
start — no last-event encoding):

- `sess_num_sessions`, `sess_events_per_session`, `sess_min_to_second_session`,
  `sess_mean_intra_gap_s` (sessionization).
- `sess_total_engagement_sec`, `sess_engagement_first_1h_sec`,
  `sess_mean_engagement_msec` (from `engagement_time_msec` — absent in v1/v2).
- `sess_num_distinct_screens` (from `firebase_screen_class`).
- Derived: `sess_returned` (≥2 sessions), `sess_engagement_early_frac`.

**Leakage check: PASS** — max directed univariate AUC 0.654
(`sess_total_engagement_sec`); none ≥ 0.85.

## Results — feature set × architecture (CV PR-AUC, 5-fold OOF, 95% CI)

| model | v2 (22 feat) | v3 (32 feat) | Δ |
|---|---|---|---|
| XGBoost (fixed params) | 0.449 [0.423, 0.475] | 0.502 [0.476, 0.528] | +0.054 |
| LightGBM | 0.463 [0.436, 0.489] | 0.538 [0.511, 0.565] | +0.076 |
| HistGradientBoosting | 0.454 [0.428, 0.482] | 0.527 [0.502, 0.555] | +0.074 |
| CatBoost | 0.444 [0.418, 0.470] | 0.512 [0.486, 0.538] | +0.068 |
| **XGBoost tuned on v3** | — | **0.548 [0.520, 0.575]** | — |

New champion CI [0.520, 0.575] is entirely **above** the v2 champion CI
[0.446, 0.497] (non-overlapping) → another real +0.08 lift.

## Held-out confirmation (frozen test, n=799) — `experiments/11_final_v3.py`

| | v1 | v2 | **v3** |
|---|---|---|---|
| Test PR-AUC | 0.384 | 0.460 | **0.591 [0.515, 0.667]** |
| Test ROC-AUC | 0.694 | 0.768 | **0.847** |
| Brier (calibration) | 0.243 | 0.193 | **0.139** |

**Cost-based operating point (FN = 3–4× FP), v3 champion on test:**

| C_FN/C_FP | threshold | precision | recall | savings vs best naive |
|---|---|---|---|---|
| 3 | 0.219 | 0.422 | 0.881 | **48%** (v2: 34%) |
| 4 | 0.124 | 0.412 | 0.962 | **54%** (v2: 36%) |

Because the model is now well calibrated (Brier 0.139), the empirical cost-optimal
threshold (~0.12–0.22) lands near the theoretical 1/(1+c) — and yields a genuinely
useful operating point: ~88–96% recall at ~41% precision.

## Cumulative progression

| iteration | features | CV PR-AUC | test PR-AUC | test ROC-AUC | cost savings |
|---|---|---|---|---|---|
| v1 (counts) | 11 | 0.400 | 0.384 | 0.694 | 21–24% |
| v2 (timing/velocity) | 22 | 0.471 | 0.460 | 0.768 | 34–36% |
| **v3 (sessions/engagement)** | 32 | **0.548** | **0.591** | **0.847** | **48–54%** |

## Conclusion

Session-level features were the **largest single lift on the test set**
(PR-AUC 0.460 → 0.591, ROC-AUC 0.768 → 0.847) — `engagement_time_msec` and
gap-based sessionization capture day-1 churn behavior that raw event counts miss.
Each iteration cleared the non-overlapping-CI bar, and the model is now a
genuinely useful retention tool: ~48–54% cost savings at the cost-optimal point.

**New model of record: XGBoost tuned on the v3 feature set.** Next candidates:
screen-navigation sequences (transition graphs), per-level funnel timing, and
calibrated probability outputs for treatment prioritisation.
