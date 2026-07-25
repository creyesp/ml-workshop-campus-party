# Markov Sequence-ORDER Features v5 — NEGATIVE Result

**Date:** 2026-06-04
**Builds on:** `2026-06-04-feature-engineering-v4-navigation.md` (v4 counts: REJECT) and the v3 champion (XGBoost, CV PR-AUC 0.548 [0.520, 0.575])
**Query:** `data/queries/flood_it_features_v5_markov.sql` · **Data:** `data/users_features_v5_markov.csv` (sha256 `07fb27e1…4c90`)
**Experiment:** `experiments/14_features_v5_markov.py`

## Hypothesis

v4 ruled out raw navigation **counts**. The open question: does the transition
**ORDER** carry orthogonal churn signal that counts cannot — i.e. first-order
**Markov transition probabilities** P(cur | prev) (which screen *follows* which,
normalized within source category) plus a few 2-gram presence flags? This is the
cheap falsification test that gates whether a sequence DNN is worth pursuing.

## What was built

Reusing the **exact same** 9 logical-screen-category mapping and 24h window guard
as v4, the query counts directed transition edges (prev_cat → cur_cat) per user
and emits the ~15 most frequent/meaningful edges as named counts plus per-source
denominators (`mk_from_game`, `mk_from_game_over`, …) and `mk_total_transitions`.

`load_v5` (on top of `load_v3`) converts those counts into **scale-free
transition probabilities** P(cur|prev) = count(prev>cur) / count(prev>\*),
normalized within each source category, and adds 4 two-gram presence flags
(`game→game_over`, `game_over→game`, `steps→shop`, `game→shop`). Users with no
transitions out of a source category get 0 for every edge from it.

**19 features added (15 probabilities + 4 flags); v3 32 → v5 51.** Two edges are
structurally all-zero in this export (`p_game_over__shop`, `p_steps__game`) — no
such transition ever occurred — and contribute nothing, consistent with the
result. Loader unit tests (`tests/data/test_load_v5.py`, 6 tests) verify split
integrity, no nulls, probabilities in [0,1], outgoing-prob sum ≤ 1, binary flags,
and zero-fill for transition-less users.

## Leakage check — PASS

Directed univariate AUC on each of the 19 new Markov features; flag threshold 0.85.

| feature | directed AUC |
|---|---|
| p_menu__level_select | 0.583 |
| p_level_select__game | 0.582 |
| ng_game_over_to_game | 0.560 |
| p_game_over__game | 0.557 |
| ng_steps_to_shop | 0.551 |
| p_steps__shop | 0.551 |
| p_game__level_select | 0.533 |
| p_game__game_over | 0.528 |
| ng_game_to_game_over | 0.528 |
| p_game__steps | 0.523 |
| p_game__game | 0.519 |
| p_game__menu | 0.510 |
| p_menu__game | 0.504 |
| p_game_over__level_select | 0.503 |
| p_game_over__menu | 0.502 |
| p_game__shop | 0.500 |
| ng_game_to_shop | 0.500 |
| p_game_over__shop | 0.500 |
| p_steps__game | 0.500 |

**Max directed AUC 0.583 — none ≥ 0.85. No leakage.** (Also no individually
strong feature: the most predictive edge is barely above chance.)

## Result — NO improvement (CV PR-AUC, 5-fold OOF, 95% CI)

| model | v3 (32 feat) | v5 (51 feat) | Δ |
|---|---|---|---|
| XGBoost (fixed) | 0.502 [0.476, 0.528] | 0.501 [0.475, 0.529] | −0.001 |
| LightGBM | 0.538 [0.511, 0.565] | 0.547 [0.521, 0.573] | +0.008 |
| HistGradientBoosting | 0.527 [0.502, 0.555] | 0.528 [0.503, 0.555] | +0.001 |
| CatBoost | 0.512 [0.486, 0.538] | 0.511 [0.485, 0.537] | −0.001 |
| XGBoost tuned on v5 | **0.538 [0.512, 0.566]** | | |
| **v3 champion** | **0.548 [0.520, 0.575]** | | |

Across every architecture the lift is −0.001 to +0.008 — within noise, with
heavily overlapping CIs. The single largest mover (LightGBM, +0.008) still has a
CI [0.521, 0.573] that overlaps v3 almost entirely. The **tuned** v5 XGBoost
(0.538 [0.512, 0.566]) is **below** the v3 champion (0.548 [0.520, 0.575]).

**Decision rule:** v5 wins only if the tuned v5 CI is strictly above the v3
champion CI (lo > 0.575). Here lo = 0.512 < 0.575 — the CIs overlap fully.

## Verdict: REJECT. Markov / sequence-ORDER hypothesis NOT supported.

The transition-order representation carries no orthogonal churn signal over v3.
This is not torturing the data: the best point estimate is +0.008 with overlapping
CIs, and the tuned model is a tick *worse* than the champion. v3 remains the
champion of record.

## Why it failed (honest interpretation)

- **Order is already implied by what's measured.** Transition probabilities like
  `p_game_over→game` (retry after a loss) or `p_steps→shop` (out-of-steps funnel)
  are near-deterministic consequences of the app's flow once you know the *counts*
  and *engagement time* that v2/v3 already capture. Conditioning on the source
  category removes volume but leaves little the label cares about.
- **Weak univariate signal.** The most predictive single transition tops out at
  directed AUC 0.583 — barely above chance — so there is little for any model to
  exploit, with or without interactions.
- **Consistent with v4.** v4 showed the navigation *graph by volume* is redundant;
  v5 shows the *graph by order/probability* is equally redundant. Both the
  magnitude and the conditional structure of screen transitions are already
  encoded by engagement counts, session structure, and engagement time.

## Implication for the sequence-DNN question

**Do not pursue a sequence DNN for this label.** Both cheap falsification tests of
the sequence hypothesis are negative: counts (v4) and first-order order/probability
(v5) each add nothing over v3. A DNN's main advantage — learning higher-order
sequential dependencies — would have to beat a signal that is already absent at
first order, on a 7,190-row training set, with no leakage headroom. The expected
return does not justify the complexity and overfitting risk. The day-1 churn
signal is saturated by aggregate engagement / session / timing features.

## Files

- `data/queries/flood_it_features_v5_markov.sql`
- `data/users_features_v5_markov.csv` (sha256 `07fb27e1…4c90`)
- `src/floodit/data/load_v5.py`
- `tests/data/test_load_v5.py`
- `experiments/14_features_v5_markov.py`
- `floodit/config.py` (V5_* column lists + `features_v5` hash)

**Champion of record unchanged: XGBoost tuned on the v3 (session) feature set,
CV PR-AUC 0.548 [0.520, 0.575], test PR-AUC 0.591.**
