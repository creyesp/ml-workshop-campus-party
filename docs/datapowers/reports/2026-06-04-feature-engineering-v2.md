# Feature Engineering v2 + Architecture Sweep — Flood-It Churn

**Date:** 2026-06-04
**Builds on:** `2026-06-04-churn-results.md` (v1 champion: tuned XGBoost, CV PR-AUC 0.400)
**Query:** `data/queries/flood_it_features_v2.sql` · **Data:** `data/users_features_v2.csv` (sha256 `d07811a8…5aa2`)
**Experiments:** `experiments/08_features_v2.py` (sweep), `experiments/09_final_v2.py` (test)

## What changed

New BigQuery-derived features, joined onto the **same** train/test split by
`user_pseudo_id` (so the comparison isolates the feature effect). All aggregated
strictly within the 24 h observation window and **anchored to the window start**
(early velocity, time-to-first-event, variety) — deliberately avoiding any
feature that encodes the *last* event time, which would leak the label.

- Raw: `cnt_events_total`, `cnt_events_first_1h`, `cnt_events_first_6h`,
  `num_distinct_event_types`, `min_to_first_{level_start,post_score,level_complete}`.
- Derived: `completion_rate`, `reset_rate`, `early_fraction_1h`, `events_per_type`.
- Dropped `num_sessions` (all-zero: `ga_session_id` absent in this export).
- `min_to_first_*` nulls ("never in window") imputed to 1441 (post-window sentinel).

**Leakage check (H3 on new features): PASS** — max directed univariate AUC 0.670
(`cnt_events_first_6h`); none ≥ 0.85. No new feature reconstructs the label.

## Results — feature set × architecture (CV PR-AUC, 5-fold OOF, 95% CI)

| model | v1 (11 feat) | v2 (22 feat) | Δ |
|---|---|---|---|
| XGBoost (v1-tuned params) | 0.399 [0.377, 0.424] | 0.449 [0.423, 0.475] | +0.050 |
| LightGBM | 0.386 [0.364, 0.410] | 0.463 [0.436, 0.489] | +0.077 |
| HistGradientBoosting | 0.400 [0.377, 0.425] | 0.454 [0.428, 0.482] | +0.054 |
| CatBoost | 0.397 [0.375, 0.422] | 0.444 [0.418, 0.470] | +0.047 |
| **XGBoost tuned on v2** | — | **0.471 [0.446, 0.497]** | — |

- **Features beat architecture.** The v2 feature lift (+0.05 to +0.08) is larger
  than the spread across architectures on v2 (0.444–0.471). LightGBM at near-default
  settings (0.463) almost matches a 30-trial-tuned XGBoost (0.471).
- **New champion: XGBoost tuned on v2 — CV PR-AUC 0.471, CI [0.446, 0.497]**, whose
  CI sits entirely **above** the v1 champion CI [0.377, 0.424] (non-overlapping) →
  a real +0.07 improvement by the spec §5 rule.

## Held-out confirmation (frozen test, n=799) — `experiments/09_final_v2.py`

| | v1 champion | v2 champion |
|---|---|---|
| Test PR-AUC | 0.384 [0.326, 0.459] | **0.460 [0.394, 0.540]** |
| Test ROC-AUC | 0.694 | **0.768** |
| Brier (calibration) | 0.243 | **0.193** |

Test CIs are wide (n=799) and overlap — as on v1, the **CV comparison is primary**
(spec §4 tripwire). Point estimate, ROC-AUC, and calibration all improve.

**Cost-based operating point (FN = 3–4× FP), v2 champion on test:**

| C_FN/C_FP | threshold | precision | recall | savings vs best naive |
|---|---|---|---|---|
| 3 | 0.518 | 0.402 | 0.665 | **34%** (was 24% with v1) |
| 4 | 0.440 | 0.385 | 0.778 | **36%** (was 21% with v1) |

## Conclusion

**Yes — feature engineering was worth it, and it was the higher-leverage move.**
The timing/velocity/variety features lifted PR-AUC 0.400 → 0.471 (CV,
non-overlapping CIs), improved out-of-sample ROC-AUC (0.694 → 0.768) and
calibration (Brier 0.243 → 0.193), and raised cost savings from ~21–24% to
~34–36%. The architecture choice mattered far less than the features.

**New model of record: XGBoost tuned on the v2 feature set**, operated at the
cost-optimal threshold (~0.44–0.52). Next candidate lifts to try: session-level
features from a properly populated `ga_session_id` source, and funnel-stage
timing — both require richer event data than this export provides.
