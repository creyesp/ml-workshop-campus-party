-- Engineered features v2 for the Flood-It churn model.
-- Keyed by user_pseudo_id so it joins onto the EXISTING train/test split
-- (same rows, same labels) — this isolates the effect of the new features.
--
-- LEAKAGE GUARD: the label is churned = (last_engagement < first+24h). Every
-- feature here is aggregated strictly within the observation window
-- (ts <= first + 24h) and is anchored to the START of the window (early
-- velocity, time-to-first-event, session/variety counts). NO feature encodes
-- the time of the LAST event, which would leak the label.
WITH
user_window AS (
  SELECT
    user_pseudo_id,
    TIMESTAMP_MICROS(MIN(event_timestamp)) AS first_ts
  FROM `firebase-public-project.analytics_153293282.events_*`
  WHERE event_name = 'user_engagement'
  GROUP BY user_pseudo_id
),
events_in_window AS (
  SELECT
    e.user_pseudo_id,
    e.event_name,
    TIMESTAMP_DIFF(TIMESTAMP_MICROS(e.event_timestamp), uw.first_ts, MINUTE) AS min_offset,
    (SELECT ep.value.int_value FROM UNNEST(e.event_params) ep
       WHERE ep.key = 'ga_session_id') AS ga_session_id
  FROM `firebase-public-project.analytics_153293282.events_*` e
  JOIN user_window uw USING (user_pseudo_id)
  WHERE TIMESTAMP_MICROS(e.event_timestamp)
        <= TIMESTAMP_ADD(uw.first_ts, INTERVAL 24 HOUR)
)
SELECT
  user_pseudo_id,
  COUNT(DISTINCT ga_session_id)                              AS num_sessions,
  COUNT(*)                                                   AS cnt_events_total,
  COUNTIF(min_offset <= 60)                                  AS cnt_events_first_1h,
  COUNTIF(min_offset <= 360)                                 AS cnt_events_first_6h,
  COUNT(DISTINCT event_name)                                 AS num_distinct_event_types,
  MIN(IF(event_name = 'level_start_quickplay',    min_offset, NULL)) AS min_to_first_level_start,
  MIN(IF(event_name = 'post_score',               min_offset, NULL)) AS min_to_first_post_score,
  MIN(IF(event_name = 'level_complete_quickplay', min_offset, NULL)) AS min_to_first_level_complete
FROM events_in_window
GROUP BY user_pseudo_id
