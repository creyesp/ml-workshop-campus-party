-- Session-level features v3 for the Flood-It churn model.
-- This 2018 Firebase export has NO ga_session_id, so sessions are derived by
-- time-gap sessionization (a new session starts after a >30 min idle gap).
-- Keyed by user_pseudo_id to join onto the existing split (same rows/labels).
--
-- LEAKAGE GUARD: all aggregates are within the 24h window and anchored to its
-- START (num sessions, time-to-2nd-session, early engagement). NO feature
-- encodes the time of the LAST event, which would leak churned = last<24h.
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
    TIMESTAMP_MICROS(e.event_timestamp) AS ts,
    uw.first_ts,
    TIMESTAMP_DIFF(TIMESTAMP_MICROS(e.event_timestamp), uw.first_ts, MINUTE) AS min_offset,
    (SELECT ep.value.int_value FROM UNNEST(e.event_params) ep
       WHERE ep.key = 'engagement_time_msec') AS eng_msec,
    (SELECT ep.value.string_value FROM UNNEST(e.event_params) ep
       WHERE ep.key = 'firebase_screen_class') AS screen
  FROM `firebase-public-project.analytics_153293282.events_*` e
  JOIN user_window uw USING (user_pseudo_id)
  WHERE TIMESTAMP_MICROS(e.event_timestamp)
        <= TIMESTAMP_ADD(uw.first_ts, INTERVAL 24 HOUR)
),
gapped AS (
  SELECT
    *,
    TIMESTAMP_DIFF(ts, LAG(ts) OVER (PARTITION BY user_pseudo_id ORDER BY ts), SECOND) AS gap_s
  FROM ev
),
sessionized AS (
  SELECT
    *,
    SUM(CASE WHEN gap_s IS NULL OR gap_s > 1800 THEN 1 ELSE 0 END)
      OVER (PARTITION BY user_pseudo_id ORDER BY ts) AS session_idx
  FROM gapped
)
SELECT
  user_pseudo_id,
  MAX(session_idx)                                            AS sess_num_sessions,
  COUNT(*) / MAX(session_idx)                                 AS sess_events_per_session,
  SUM(IFNULL(eng_msec, 0)) / 1000.0                           AS sess_total_engagement_sec,
  SUM(IF(min_offset <= 60, IFNULL(eng_msec, 0), 0)) / 1000.0  AS sess_engagement_first_1h_sec,
  AVG(eng_msec)                                               AS sess_mean_engagement_msec,
  COUNT(DISTINCT screen)                                      AS sess_num_distinct_screens,
  AVG(IF(gap_s <= 1800, gap_s, NULL))                         AS sess_mean_intra_gap_s,
  MIN(IF(session_idx = 2, min_offset, NULL))                  AS sess_min_to_second_session
FROM sessionized
GROUP BY user_pseudo_id
