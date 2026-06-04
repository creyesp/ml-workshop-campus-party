# Difficulty / Progression Features v6 — NEGATIVE Result

**Date:** 2026-06-04
**Builds on:** v3 champion (XGBoost, CV PR-AUC 0.548 [0.520, 0.575])
**Query:** `data/queries/flood_it_features_v6_difficulty.sql` · **Data:** `data/users_features_v6_difficulty.csv` (sha256 `29715db6…3fe8`)
**Experiment:** `experiments/15_features_v6.py`

## Hypothesis

Difficulty/progression signals — how hard the user found the game and how far
they got — drive day-1 churn (a user stuck failing early levels churns).

## What was built (difficulty data DOES exist)

The export carries difficulty/progression params absent from v1–v5:
- `level_fail_quickplay` / `level_fail` and `level_retry_quickplay` / `level_retry`
  events → **fail / retry counts** (frustration; the base set only had `reset`).
- `level` param (range 0–31) → **max level reached**, distinct levels.
- `score` param → **max / mean score**.
- (`time` param is ~all-zero in this export — unusable; timing is already covered
  by v3 `engagement_time`.)

11 features added (v3 32 → v6 43): fail/retry/level_up counts, max_level,
distinct_levels, max/mean score, score count, and ratios (fail_rate, retry_rate,
fail_to_complete). max_level/score nulls imputed to 0.

**Leakage check: PASS** (max directed AUC 0.620, `diff_num_distinct_levels`).

## Result — NO improvement (CV PR-AUC, 5-fold OOF, 95% CI)

| model | v3 (32 feat) | v6 (43 feat) | Δ |
|---|---|---|---|
| XGBoost (fixed) | 0.502 [0.476, 0.528] | 0.505 [0.479, 0.532] | +0.003 |
| LightGBM | 0.538 [0.511, 0.565] | 0.539 [0.513, 0.566] | +0.001 |
| HistGradientBoosting | 0.527 [0.502, 0.555] | 0.532 [0.506, 0.558] | +0.004 |
| CatBoost | 0.512 [0.486, 0.538] | 0.519 [0.493, 0.545] | +0.007 |
| XGBoost tuned on v6 | **0.547 [0.521, 0.575]** | | |
| **v3 champion** | **0.548 [0.520, 0.575]** | | |

Lift is +0.00 to +0.01 across every architecture; the tuned v6 model (0.547) is a
hair *below* the v3 champion (0.548) with essentially identical CIs.
**Verdict: REJECT v6. v3 remains champion.**

## Why it failed

Difficulty counts (fails, retries, levels) are **collinear with engagement
volume** already captured by v3: a user who fails/retries a lot is a user who
played a lot (more events, longer engagement). The difficulty signal is not
orthogonal to "how much did they engage."

## Where this leaves feature engineering

This is the **third consecutive negative behavioral iteration** (v4 navigation,
v5 sequence-order/Markov, v6 difficulty) — all consistent with the
**learning-curve finding** that the v3 model has plateaued (`learning-curve.md`).
The v3 engagement/session feature set has effectively **saturated the learnable
day-1 churn signal** from in-app behavior; the test ceiling (~0.59 PR-AUC) is set
by the problem's inherent predictability, not by missing behavioral features.

**Recommendation: stop behavioral feature engineering.** The remaining levers are
*non-behavioral* and out of the current data's scope — acquisition source /
campaign, device/network quality, first-session crash/error signals — or simply
**ship the v3 champion**, which already delivers ~48–54% retention-cost savings.

**Champion of record unchanged: XGBoost on v3 (session) features — CV PR-AUC
0.548, test 0.591, ROC-AUC 0.847.**
