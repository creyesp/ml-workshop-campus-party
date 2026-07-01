# Navigation-Sequence Features v4 — NEGATIVE Result

**Date:** 2026-06-04
**Builds on:** `2026-06-04-feature-engineering-v3-sessions.md` (v3 champion: XGBoost, CV PR-AUC 0.548)
**Query:** `data/queries/flood_it_features_v4_navigation.sql` · **Data:** `data/users_features_v4_navigation.csv` (sha256 `d652d104…6f6e`)
**Experiment:** `experiments/12_features_v4.py`

## Hypothesis

Screen-to-screen navigation patterns (transition graph) capture day-1 churn
behavior that aggregate counts miss — e.g. bouncing off `game_over`, reaching
`shop`, hitting `out_of_steps`, navigation breadth/revisits.

## What was built

`firebase_screen_class` / `firebase_previous_class` are platform-specific
(Android `game_board` vs iOS `FIGameViewController`, etc.), so each was mapped to
one of 9 **logical categories** (game, game_over, menu, shop, steps,
level_select, ad, info, root). Per user within the 24 h window: screen-view
count, distinct categories, total/distinct transitions, self-loops, and visit
counts to game_over / shop / steps / ad / level_select, plus derived rates
(transition rate, revisit rate, game_over rate, reached_shop, reached_steps).
15 features added (v3 32 → v4 47). 30 train users with no screen views → zero-filled.

**Leakage check: PASS** (max directed AUC 0.659, `nav_screen_views`).

## Result — NO improvement (CV PR-AUC, 5-fold OOF, 95% CI)

| model | v3 (32 feat) | v4 (47 feat) | Δ |
|---|---|---|---|
| XGBoost (fixed) | 0.502 [0.476, 0.528] | 0.505 [0.480, 0.531] | +0.002 |
| LightGBM | 0.538 [0.511, 0.565] | 0.538 [0.511, 0.564] | −0.000 |
| HistGradientBoosting | 0.527 [0.502, 0.555] | 0.528 [0.502, 0.556] | +0.001 |
| CatBoost | 0.512 [0.486, 0.538] | 0.520 [0.493, 0.547] | +0.008 |
| XGBoost tuned on v4 | **0.542 [0.515, 0.568]** | | |
| **v3 champion** | **0.548 [0.520, 0.575]** | | |

Across every architecture the lift is +0.00 to +0.01 — within noise. The tuned v4
model (0.542) is actually **below** the v3 champion (0.548), with heavily
overlapping CIs. **Verdict: REJECT v4. v3 remains the champion.**

## Why it failed (what was ruled out)

The navigation signal is **redundant** with features v2/v3 already capture:
- `nav_screen_views`, `nav_distinct_categories`, `nav_transitions` are collinear
  with `cnt_events_total`, `num_distinct_event_types`, and the session counts.
- `nav_cnt_game_over` tracks the existing `cnt_level_*` game-progress counts.
- Engagement *time* (v3) already encodes depth-of-play better than screen hops.

So the screen-transition graph adds no orthogonal information for this label.

## Takeaways

- **Don't pursue** raw screen-navigation counts further — ruled out.
- **What to try instead** (orthogonal signal, not volume): sequence *order* models
  (e.g. n-gram / Markov transition probabilities as features, not just counts),
  inter-event timing within sessions, and per-level difficulty/funnel timing.
- Process win: the non-overlapping-CI gate did its job — it stopped a
  plausible-but-null feature set from being declared a winner on a +0.008 point
  estimate.

**Champion of record unchanged: XGBoost tuned on the v3 (session) feature set,
CV PR-AUC 0.548, test PR-AUC 0.591.**
