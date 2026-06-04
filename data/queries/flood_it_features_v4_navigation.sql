-- Navigation-sequence features v4 for the Flood-It churn model.
-- Screen classes are platform-specific (Android vs iOS names for the same
-- logical screen), so each firebase_screen_class / firebase_previous_class is
-- mapped to a platform-agnostic logical category, then aggregated per user.
-- Keyed by user_pseudo_id to join onto the existing split (same rows/labels).
--
-- LEAKAGE GUARD: all aggregates are within the 24h window. They count screen
-- visits / transitions by type and breadth; none encodes the LAST event time.
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
)
SELECT
  user_pseudo_id,
  COUNT(*)                                                AS nav_screen_views,
  COUNT(DISTINCT cur_cat)                                 AS nav_distinct_categories,
  COUNTIF(prev_cat IS NOT NULL)                           AS nav_transitions,
  COUNT(DISTINCT IF(prev_cat IS NOT NULL, CONCAT(prev_cat, '>', cur_cat), NULL)) AS nav_distinct_transitions,
  COUNTIF(prev_cat IS NOT NULL AND prev_cat = cur_cat)    AS nav_self_loops,
  COUNTIF(cur_cat = 'game_over')                          AS nav_cnt_game_over,
  COUNTIF(cur_cat = 'shop')                               AS nav_cnt_shop,
  COUNTIF(cur_cat = 'steps')                              AS nav_cnt_steps,
  COUNTIF(cur_cat = 'ad')                                 AS nav_cnt_ad,
  COUNTIF(cur_cat = 'level_select')                       AS nav_cnt_level_select
FROM mapped
GROUP BY user_pseudo_id
