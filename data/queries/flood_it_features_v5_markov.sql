-- Markov / sequence-ORDER features v5 for the Flood-It churn model.
-- The OPEN QUESTION after v4: raw navigation COUNTS added nothing. Does the
-- transition ORDER (first-order Markov screen-transition structure) carry
-- orthogonal signal? This query emits counts for the ~15 most frequent/meaningful
-- directed transition edges (prev_cat -> cur_cat), which the feature builder then
-- normalizes into per-source transition PROBABILITIES P(cur|prev).
--
-- Uses the EXACT SAME logical-screen-category mapping as
-- data/queries/flood_it_features_v4_navigation.sql (9 categories) and the SAME
-- 24h window guard. Keyed by user_pseudo_id to join onto the existing split.
--
-- LEAKAGE GUARD: all transitions are within the 24h window. Edge counts describe
-- WHICH screen followed WHICH within the window; none encodes the LAST event time
-- (which would leak churned = last_engagement < 24h).
WITH
user_window AS (
  SELECT
    user_pseudo_id,
    TIMESTAMP_MICROS(MIN(event_timestamp)) AS first_ts
  FROM `firebase-public-project.analytics_153293282.events_*`
  WHERE event_name = 'user_engagement'
  GROUP BY user_pseudo_id
),
mapped AS (
  SELECT
    e.user_pseudo_id,
    cat.cur AS cur_cat,
    cat.prev AS prev_cat
  FROM `firebase-public-project.analytics_153293282.events_*` e
  JOIN user_window uw USING (user_pseudo_id),
  UNNEST([STRUCT(
    (SELECT ep.value.string_value FROM UNNEST(e.event_params) ep WHERE ep.key='firebase_screen_class') AS raw_cur,
    (SELECT ep.value.string_value FROM UNNEST(e.event_params) ep WHERE ep.key='firebase_previous_class') AS raw_prev
  )]) AS s,
  UNNEST([STRUCT(
    CASE
      WHEN s.raw_cur IN ('game_board','FloodItActivity','FIGameViewController') THEN 'game'
      WHEN s.raw_cur IN ('game_over','GameFinishedActivity','FIGameOverViewController') THEN 'game_over'
      WHEN s.raw_cur IN ('main_menu','MainMenuActivity','FIMainMenuViewController') THEN 'menu'
      WHEN s.raw_cur IN ('shop_menu','iap','FIGetMoreStepsTableViewController') THEN 'shop'
      WHEN s.raw_cur IN ('out_of_steps','extra_steps','ExtraStepsActivity','FIOutOfStepsViewController') THEN 'steps'
      WHEN s.raw_cur IN ('level_select','LevelSelectionActivity','FILevelsCollectionViewController') THEN 'level_select'
      WHEN REGEXP_CONTAINS(s.raw_cur, r'(?i)(^ad|gad|mraid|^mp|interstitial|browser|safari)') THEN 'ad'
      WHEN s.raw_cur IN ('settings','SettingsActivity','AboutActivity','how_to_play','FIOptionsViewController','FIHowToPlayViewController','stats','FIStatisticsViewController') THEN 'info'
      WHEN s.raw_cur = 'FIRootViewController' THEN 'root'
      WHEN s.raw_cur IS NULL THEN NULL
      ELSE 'other'
    END AS cur,
    CASE
      WHEN s.raw_prev IN ('game_board','FloodItActivity','FIGameViewController') THEN 'game'
      WHEN s.raw_prev IN ('game_over','GameFinishedActivity','FIGameOverViewController') THEN 'game_over'
      WHEN s.raw_prev IN ('main_menu','MainMenuActivity','FIMainMenuViewController') THEN 'menu'
      WHEN s.raw_prev IN ('shop_menu','iap','FIGetMoreStepsTableViewController') THEN 'shop'
      WHEN s.raw_prev IN ('out_of_steps','extra_steps','ExtraStepsActivity','FIOutOfStepsViewController') THEN 'steps'
      WHEN s.raw_prev IN ('level_select','LevelSelectionActivity','FILevelsCollectionViewController') THEN 'level_select'
      WHEN REGEXP_CONTAINS(s.raw_prev, r'(?i)(^ad|gad|mraid|^mp|interstitial|browser|safari)') THEN 'ad'
      WHEN s.raw_prev IN ('settings','SettingsActivity','AboutActivity','how_to_play','FIOptionsViewController','FIHowToPlayViewController','stats','FIStatisticsViewController') THEN 'info'
      WHEN s.raw_prev = 'FIRootViewController' THEN 'root'
      WHEN s.raw_prev IS NULL THEN NULL
      ELSE 'other'
    END AS prev
  )]) AS cat
  WHERE TIMESTAMP_MICROS(e.event_timestamp) <= TIMESTAMP_ADD(uw.first_ts, INTERVAL 24 HOUR)
    AND cat.cur IS NOT NULL
    AND cat.prev IS NOT NULL
)
SELECT
  user_pseudo_id,
  COUNT(*)                                                       AS mk_total_transitions,
  -- source-category totals (denominators for P(cur|prev) in the feature builder)
  COUNTIF(prev_cat = 'game')                                     AS mk_from_game,
  COUNTIF(prev_cat = 'game_over')                                AS mk_from_game_over,
  COUNTIF(prev_cat = 'menu')                                     AS mk_from_menu,
  COUNTIF(prev_cat = 'level_select')                             AS mk_from_level_select,
  COUNTIF(prev_cat = 'steps')                                    AS mk_from_steps,
  -- ~15 most meaningful / among-most-frequent directed edges (prev>cur)
  COUNTIF(prev_cat = 'game'         AND cur_cat = 'game')        AS mk_game__game,
  COUNTIF(prev_cat = 'game'         AND cur_cat = 'game_over')   AS mk_game__game_over,
  COUNTIF(prev_cat = 'game'         AND cur_cat = 'menu')        AS mk_game__menu,
  COUNTIF(prev_cat = 'game'         AND cur_cat = 'steps')       AS mk_game__steps,
  COUNTIF(prev_cat = 'game'         AND cur_cat = 'shop')        AS mk_game__shop,
  COUNTIF(prev_cat = 'game'         AND cur_cat = 'level_select')AS mk_game__level_select,
  COUNTIF(prev_cat = 'game_over'    AND cur_cat = 'game')        AS mk_game_over__game,
  COUNTIF(prev_cat = 'game_over'    AND cur_cat = 'menu')        AS mk_game_over__menu,
  COUNTIF(prev_cat = 'game_over'    AND cur_cat = 'level_select')AS mk_game_over__level_select,
  COUNTIF(prev_cat = 'game_over'    AND cur_cat = 'shop')        AS mk_game_over__shop,
  COUNTIF(prev_cat = 'menu'         AND cur_cat = 'game')        AS mk_menu__game,
  COUNTIF(prev_cat = 'menu'         AND cur_cat = 'level_select')AS mk_menu__level_select,
  COUNTIF(prev_cat = 'level_select' AND cur_cat = 'game')        AS mk_level_select__game,
  COUNTIF(prev_cat = 'steps'        AND cur_cat = 'shop')        AS mk_steps__shop,
  COUNTIF(prev_cat = 'steps'        AND cur_cat = 'game')        AS mk_steps__game
FROM mapped
GROUP BY user_pseudo_id
