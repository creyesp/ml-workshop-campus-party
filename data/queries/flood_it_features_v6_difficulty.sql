-- Difficulty / progression features v6 for the Flood-It churn model.
-- Captures NEW difficulty signal absent from v1-v5: level FAIL / RETRY events
-- (frustration), max level reached (progression depth), and score performance.
-- Keyed by user_pseudo_id to join onto the existing split (same rows/labels).
--
-- LEAKAGE GUARD: all aggregates are within the 24h window and reflect
-- in-window progress/difficulty; none encodes the LAST event time.
WITH
user_window AS (
  SELECT
    user_pseudo_id,
    TIMESTAMP_MICROS(MIN(event_timestamp)) AS first_ts
  FROM `firebase-public-project.analytics_153293282.events_*`
  WHERE event_name = 'user_engagement'
  GROUP BY user_pseudo_id
),
ev AS (
  SELECT
    e.user_pseudo_id,
    e.event_name,
    (SELECT COALESCE(ep.value.int_value, CAST(ep.value.double_value AS INT64))
       FROM UNNEST(e.event_params) ep WHERE ep.key = 'level') AS level,
    (SELECT COALESCE(ep.value.int_value, CAST(ep.value.double_value AS INT64))
       FROM UNNEST(e.event_params) ep WHERE ep.key = 'score') AS score
  FROM `firebase-public-project.analytics_153293282.events_*` e
  JOIN user_window uw USING (user_pseudo_id)
  WHERE TIMESTAMP_MICROS(e.event_timestamp)
        <= TIMESTAMP_ADD(uw.first_ts, INTERVAL 24 HOUR)
)
SELECT
  user_pseudo_id,
  COUNTIF(event_name IN ('level_fail_quickplay', 'level_fail'))   AS diff_cnt_fail,
  COUNTIF(event_name IN ('level_retry_quickplay', 'level_retry'))  AS diff_cnt_retry,
  COUNTIF(event_name = 'level_up')                                 AS diff_cnt_level_up,
  MAX(level)                                                       AS diff_max_level,
  COUNT(DISTINCT IF(level IS NOT NULL, level, NULL))               AS diff_num_distinct_levels,
  MAX(score)                                                       AS diff_max_score,
  AVG(score)                                                       AS diff_mean_score,
  COUNTIF(event_name = 'post_score')                               AS diff_cnt_scores
FROM ev
GROUP BY user_pseudo_id
