# Leakage Audit (H3) — Flood-It Churn

**Date:** 2026-06-04
**Spec:** §7 H3, §8 (window leakage) · `experiments/01_leakage_audit.py`
**Data:** `users_train.csv` (sha256 `1dea2802…ce14`), n=7,190, prevalence 0.231

## Why audit

`churned` is derived from `user_last_engagement` (global MAX of `user_engagement`
events), while the `cnt_*` features count events with `timestamp <= first+24h`
(the observation window, which closes at prediction time — spec §3). The concern:
does any `cnt_*` nevertheless encode the post-window label? `cnt_user_engagement`
is the prime suspect, since engagement and the churn definition both ride on
`user_engagement` events.

## Method

- **Univariate (directed) ROC-AUC:** each feature used alone as a score; folded
  to [0.5, 1]. Near 1.0 ⇒ the feature almost determines the label.
- **Ablation:** 5-fold stratified CV PR-AUC with the feature neutralised
  (set constant ⇒ StandardScaler emits zeros) vs. the full model. A large drop ⇒
  the model leans on that one feature.
- Model: class-balanced Logistic Regression on the standard preprocessor.

## Results

| feature | directed AUC | CV PR-AUC full | CV PR-AUC ablated | Δ |
|---|---|---|---|---|
| cnt_user_engagement | 0.676 | 0.321 | 0.315 | 0.005 |
| cnt_post_score | 0.582 | 0.321 | 0.321 | 0.000 |
| cnt_spend_virtual_currency | 0.554 | 0.321 | 0.320 | 0.000 |
| cnt_use_extra_steps | 0.550 | 0.321 | 0.320 | 0.000 |
| cnt_level_reset_quickplay | 0.547 | 0.321 | 0.321 | −0.000 |
| cnt_level_start_quickplay | 0.547 | 0.321 | 0.321 | −0.001 |
| cnt_completed_5_levels | 0.540 | 0.321 | 0.318 | 0.003 |
| cnt_level_end_quickplay | 0.527 | 0.321 | 0.320 | 0.000 |
| cnt_ad_reward | 0.504 | 0.321 | 0.321 | −0.001 |
| cnt_challenge_a_friend | 0.504 | 0.321 | 0.319 | 0.002 |
| cnt_level_complete_quickplay | 0.502 | 0.321 | 0.322 | −0.001 |

Tripwire thresholds: directed AUC ≥ 0.85, or ablated PR-AUC ≤ prevalence+0.02
(≤ 0.251) with Δ > 0.10.

## Verdict: **PASS — no disqualifying leakage**

- Max directed AUC is 0.676 (`cnt_user_engagement`) — far below 0.85. No feature
  comes close to determining the label.
- No single-feature ablation collapses the model: the largest Δ is 0.005, and the
  ablated PR-AUC never falls to the prevalence floor. The model's ~0.32 PR-AUC is
  **distributed across features**, not propped up by one leaky column.
- `cnt_user_engagement` is the strongest single signal (directed AUC 0.676) but
  does not reconstruct the label. Note a **truncation asymmetry**: churned users
  are counted over their whole (sub-24h) lifetime while retained users are
  counted only over their first-24h slice, so churned users actually show a
  *higher* mean count (37.4 vs 28.4). This is an honest within-window effect, not
  post-window leakage — the count is time-boxed to the observation window (spec
  §3) and is not a deterministic function of the label.

**Conclusion:** H3 holds — the `cnt_*` features do not leak post-prediction-time
information. The feature set is cleared; proceed to baselines (Task 10). No
features quarantined.
