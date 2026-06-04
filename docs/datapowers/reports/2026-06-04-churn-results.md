# Flood-It Churn Model — Improvement Results & Decision

**Date:** 2026-06-04
**Spec:** `docs/datapowers/specs/2026-06-04-churn-model-improvement-spec.md`
**Plan:** `docs/datapowers/plans/2026-06-04-churn-model-improvement.md`
**Code:** `src/floodit/` · **Experiments:** `experiments/` · **Tracking:** MLflow (`mlruns/`, sqlite)
**Data:** `users_train.csv` (sha256 `1dea2802…ce14`, n=7,190, 23.1% churn) · `users_test.csv` (sha256 `57a5adf5…23bb`, n=799, 23.2% churn)

## TL;DR

The tuned XGBoost is the better model — it beats the strong Logistic-Regression
baseline on **cross-validated PR-AUC with non-overlapping CIs** (the primary,
higher-powered comparison) and passes the leakage audit. **But** the spec's
placeholder operating constraint (precision ≥ 0.60) is **infeasible at usable
recall** for either model, so this is a **conditional adopt / iterate**, not an
unqualified deploy: take XGBoost as the new model of record, but set a realistic
operating point with the growth team before any deployment.

## Metrics

| Comparison | Logistic Regression (strong baseline) | XGBoost (tuned champion) |
|---|---|---|
| **CV PR-AUC** (train, 5-fold OOF) | 0.315, CI **[0.297, 0.336]** | **0.400, CI [0.377, 0.424]** |
| CV ROC-AUC | 0.622 | 0.724 |
| Test PR-AUC (frozen, n=799) | 0.295, CI [0.253, 0.351] | 0.384, CI [0.326, 0.459] |
| Test ROC-AUC / Brier | 0.614 / 0.243 | 0.694 / 0.243 |
| Trivial floor (majority) | PR-AUC 0.231 (= prevalence), ROC-AUC 0.500 | — |

- **CV (n=7,190):** champion CI [0.377, 0.424] does **not** overlap baseline CI
  [0.297, 0.336] → champion wins by the spec §5 non-overlapping-CI rule.
- **Test (n=799):** CIs **overlap** ([0.326, 0.459] vs [0.253, 0.351]). Per the
  spec §4 tripwire ("test too small to separate with non-overlapping CIs →
  escalate to a CV-only comparison and flag the limitation"), the **CV comparison
  is primary** and the point-estimate test win is reported with that caveat.

## Hypotheses

- **H3 — No window leakage: CONFIRMED (PASS).** Max directed univariate ROC-AUC
  is 0.676 (`cnt_user_engagement`), far below the 0.85 tripwire; no single-feature
  ablation collapses the model. Features are time-boxed to the observation window
  (spec §3) and do not reconstruct the label. Independently re-derived and
  approved. See `2026-06-04-leakage-audit.md`. No features quarantined.
- **H2 — Tuned XGBoost beats linear baseline (non-overlapping CIs): SUPPORTED**
  on CV (primary). Best params: depth 2, 300 trees, lr 0.069, subsample 0.87,
  colsample 0.91, min_child_weight 3, reg_lambda 4.43, scale_pos_weight 4.12
  (`experiments/_artifacts/xgb_best_params.json`).
- **H1 — Cost-tuned threshold improves recall @ precision ≥ 0.60: REJECTED.** The
  precision ≥ 0.60 target is **unreachable at usable recall**: on train OOF the
  threshold that hits precision 0.60 is ~0.91 with recall ≈ 0.001; on test the
  champion's recall @ precision ≥ 0.60 is 0.032 (confusion matrix at that
  threshold predicts all-negative). At the default 0.5 threshold precision is only
  0.33. **Conclusion: 0.60 is the wrong precision target for this data/model**, not
  that tuning is useless — it directly motivates Q1/Q2 below.

## Robustness & fairness (visibility only)

- **Non-stationarity (Q3):** time-based split (train early cohorts → validate
  late cohorts) gives champion PR-AUC 0.499, **no degradation** vs random-CV
  (0.384). No evidence of harmful drift over calendar time; a time split need not
  replace stratified k-fold as the primary strategy.
- **Segments (spec §8, report-only):** champion test PR-AUC by `device_os` —
  iOS 0.44 (n=414) vs Android 0.30 (n=363); the small null-`device_os` group
  (n=22) has 73% prevalence. By `country_name`, large segments range US 0.40,
  India 0.29, Japan 0.59; small-n countries (≤ ~25) are noisy. Disparities are
  **visible but not formally audited** (deferred per spec §9).

## Decision (spec §10)

**ITERATE → CONDITIONAL ADOPT.**

- ✅ Champion beats the strong baseline on PR-AUC with non-overlapping **CV** CIs.
- ✅ Passes the leakage audit (H3).
- ❌ Has **no** cost-justified operating threshold meeting precision ≥ 0.60 (the
  third §10 deploy gate fails — the target is infeasible).

Because the operating-point gate fails, do **not** deploy at precision ≥ 0.60.
Adopt **XGBoost as the new model of record** (replacing the undocumented
0.5-threshold `classification_report` baseline) and resolve the operating point
with stakeholders before production. Reproducibility verified: clean-venv
rebuild, 43 tests pass, CV metrics reproduce within ±0.005.

## Open questions for the growth team (spec §11)

- **Q1 — cost ratio (FP vs FN).** This sets the real operating threshold. The
  precision ≥ 0.60 placeholder yields ~0 recall, so it is almost certainly too
  strict; a defensible threshold needs the actual cost of a wasted incentive vs a
  lost retainable user.
- **Q2 — precision/volume budget.** How many users can be treated per day? With a
  cost-justified threshold (Q1), the operating point should be reported as
  recall @ that budget, not at an arbitrary precision floor.
- **Q3 — time split as primary?** Not warranted now: no non-stationarity
  degradation was observed. Keep stratified k-fold primary; revisit if a later
  cohort shows drift in monitoring.
