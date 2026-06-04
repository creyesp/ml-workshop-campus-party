# Flood-It Churn Model — Improvement Experiment Spec

**Date:** 2026-06-04
**Author:** creyesp (ml-workshop-campus-party)
**Status:** Draft

## 1. Problem

- **Decision the model informs:** Whether to trigger a retention action (push notification, in-app offer, guided tutorial) on a *new* user during their first day, before they churn.
- **Stakeholder & consumption:** Growth / retention team. The model outputs a churn probability per new user; users above an operating threshold are enrolled in a retention treatment.
- **Why now / what changes downstream:** A churn model already exists in the workshop notebooks (LogReg, RandomForest, GradientBoosting, XGBoost + Optuna + SHAP), but it is evaluated only with a single `classification_report` + confusion matrix at the default 0.5 threshold, on one train/test split, with no leakage audit and no documented baseline. This experiment hardens that pipeline so "the model improved" becomes a defensible claim instead of a number that moved.

## 2. Target Definition

- **Operational definition (SQL, from `data/queries/flood_it_dataset.sql`):**
  ```sql
  churned = IF(user_last_engagement < TIMESTAMP_ADD(user_first_engagement, INTERVAL 24 HOUR), 1, 0)
  ```
  A user is `churned = 1` if they have **no `user_engagement` event after the first 24 h** following their first engagement.
- **Positive class:** churned (did not return after 24 h). **Negative class:** retained (engaged at or after the 24 h mark).
- **Edge cases:** Users within 10 min of first engagement (`bounced = 1`) and users too recent to observe a full 24 h window (`is_enable = 0`) are **excluded upstream** — confirmed 0% of `users_train.csv` / `users_test.csv` rows have `bounced = 1` or `is_enable = 0`. Users near the 24 h boundary are inherently ambiguous (label noise).
- **Labeling source / latency / noise:** Label derived deterministically from event timestamps in BigQuery (`firebase-public-project.analytics_153293282.events_*`). Latency = 24 h after first engagement. Noise source = the fixed 24 h cutoff is arbitrary.

## 3. Unit of Analysis & Timing

- **One row per:** `user_pseudo_id` (one new user).
- **Prediction time:** at the close of the observation window — i.e. **24 h after the user's first engagement**.
- **Label observation time:** immediately after the same 24 h window closes.
- **Features available at prediction time:** event counts (`cnt_*`) aggregated over events with `timestamp <= user_first_engagement + 24h`, plus `country_name`, `device_os`, `device_lang` (taken from the most recent `user_engagement` event).

## 4. Success Metrics

- **Primary:** **PR-AUC (Average Precision)** for the churn (positive) class on a held-out test set. Chosen over ROC-AUC because the positive class is the minority (~23%) and is the class the retention decision acts on. Threshold-agnostic, so model comparison does not depend on an arbitrary cutoff.
- **Guardrails:**
  - **ROC-AUC** reported as a secondary, threshold-agnostic comparison metric.
  - **Operating point:** decision threshold selected by business cost (not the implicit 0.5), reported as **recall of churn @ a fixed precision** (target precision ≥ 0.60, to be revisited with growth team).
  - **Calibration:** report a reliability summary (e.g. Brier score) so probabilities are usable for treatment prioritization.
- **Tripwires (stop the experiment):**
  - Leakage audit (Section 8 / H3) shows a `cnt_*` feature encodes post-prediction-time information → halt and redefine features before any further modeling.
  - Test set (n≈799) too small to separate candidate models with non-overlapping confidence intervals → escalate to a CV-only comparison and flag the limitation rather than report a point estimate.

## 5. Baseline

- **Trivial baseline:** majority-class predictor (always predict "retained"). Accuracy ≈ 0.77; PR-AUC ≈ prevalence ≈ 0.23. Any model must beat the prevalence PR-AUC floor.
- **Strong baseline:** **Logistic Regression** with class balancing on the existing preprocessing pipeline — re-scored under the new evaluation protocol (PR-AUC + CI + cross-validation). This becomes the documented number to beat, replacing the undocumented 0.5-threshold `classification_report`.
- **Score required to beat:** A candidate model "wins" only if its **PR-AUC confidence interval does not overlap** the strong baseline's CI (point-estimate improvement is insufficient).

## 6. Datasets

- **Source / owner / cadence:** `data/users_train.csv` (7,190 rows) and `data/users_test.csv` (799 rows), generated from the public Firebase BigQuery export via `data/queries/flood_it_dataset.sql`. Static workshop snapshot; no refresh cadence. No PII beyond the pseudonymous `user_pseudo_id`.
- **Version pin:** hash-pin both CSVs (and the raw `users_raw.csv`) so the experiment is reproducible; record hashes in the experiment tracker.
- **Split strategy & why:**
  - The original train/test split is reused as the **final held-out test set** for the headline number (kept frozen, scored once per candidate).
  - **Model selection / tuning** uses **k-fold cross-validation on the training set** (not the single split) to produce confidence intervals and reduce the variance of a 799-row test set.
  - Rationale: per-user rows are independent (one row per user), so stratified k-fold on `churned` is appropriate; a *time-based* split is also evaluated as a robustness check because the underlying process (cohorts over calendar time via `user_first_engagement`) may be non-stationary.
- **Sample size & class balance:** train 7,190 (23.1% churn), test 799 (23.2% churn) — balance is stable across splits.

## 7. Hypotheses

- **H1 — Threshold tuning:** Selecting the decision threshold by business cost instead of the default 0.5 will improve churn **recall @ precision ≥ 0.60** by a meaningful margin over the current 0.5-threshold operating point, because the positive class is the minority and 0.5 is not the cost-optimal cutoff. *Falsification:* if the cost-tuned threshold does not improve recall@precision over 0.5 on cross-validated folds, reject.
- **H2 — Tuned XGBoost vs. linear baseline:** A properly cross-validated, Optuna-tuned XGBoost will beat the strong (Logistic Regression) baseline on **PR-AUC with non-overlapping CIs**. *Falsification:* if XGBoost's PR-AUC CI overlaps the baseline's, it does not "win" and we keep the simpler model.
- **H3 — No window leakage:** No `cnt_*` feature leaks information observable only after prediction time. *Falsification:* if removing any single feature collapses a suspiciously high score, or a feature is shown to be a deterministic function of the label window, the leakage hypothesis is confirmed and features are redefined.

Each hypothesis is falsifiable via a check defined in Section 6 (splits/CV) or run directly.

## 8. Risks & Mitigations

| Risk | Mitigation / falsifiable test |
|---|---|
| **Window leakage** (features & label share the 24 h window) | Audit each `cnt_*` against the SQL window logic; run the `preventing-data-leakage` skill; permutation/ablation check (H3) before trusting any score. |
| **High-variance test metric** (n≈799) | Report cross-validated CIs; require non-overlapping CIs to declare a winner (Section 5). |
| **Non-stationarity** (cohorts over calendar time) | Time-based split robustness check in addition to stratified k-fold (Section 6). |
| **Reproducibility** (no pinned data/seed today) | Hash-pin CSVs, fix random seeds, log code SHA + data hash + params + metrics via `experiment-tracking`. |
| **Fairness / proxy attributes** (`country_name`, `device_os`, `device_lang`) | Out of scope for this experiment, but flagged: segment-level metric reporting by country/OS is included in evaluation so disparities are visible even though a formal audit is deferred. |

## 9. Out of Scope

- Serving / production packaging, the `serving/` echo service, and online inference.
- Drift monitoring and retraining cadence.
- Formal fairness & bias audit (segment metrics are reported, but acceptance criteria / remediation are deferred).
- New data collection or changing the 24 h target definition.
- Deep learning models (the BQML DNN in `extras/` is reference only).

## 10. Decision Criteria

- **Deploy / adopt as new champion:** candidate beats the strong baseline on PR-AUC with **non-overlapping CIs**, passes the leakage audit (H3), and has a cost-justified operating threshold meeting precision ≥ 0.60.
- **Kill:** leakage audit invalidates the feature set (tripwire), or no candidate beats the prevalence/strong baseline after tuning.
- **Iterate:** candidate improves PR-AUC point estimate but CIs overlap → revisit features (`feature-engineering-systematically`) or validation strategy rather than declaring victory.

## 11. Open Questions

- **Q1 (stakeholder):** What is the actual cost ratio of a false positive (wasted incentive) vs. false negative (lost retainable user)? This sets the cost-optimal threshold in H1; currently assumed precision ≥ 0.60 as a placeholder.
- **Q2 (stakeholder):** Is there a hard precision or volume budget for the retention campaign (how many users can be treated per day)?
- **Q3:** Should the time-based split become the primary evaluation if non-stationarity is detected, or remain a robustness check?
