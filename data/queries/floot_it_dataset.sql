-- CREATE OR REPLACE TABLE workshop.floot_it_dataset as
WITH 
user_returning AS 
(
  SELECT
    user_pseudo_id,
    user_first_engagement,
    user_last_engagement,
    EXTRACT(MONTH     FROM user_first_engagement) AS month,
    EXTRACT(DAYOFYEAR FROM user_first_engagement) AS julianday,
    EXTRACT(DAYOFWEEK FROM user_first_engagement) AS dayofweek,
    TIMESTAMP_ADD(user_first_engagement, INTERVAL 24 HOUR) ts_after_first_engagement,
    IF (user_first_engagement > TIMESTAMP_SUB(max_ts_events, INTERVAL 24*2 HOUR), 0, 1) AS is_enable,
    IF (user_last_engagement <= TIMESTAMP_ADD(user_first_engagement, INTERVAL 10 MINUTE), 1, 0) AS bounced,
    IF (user_last_engagement <  TIMESTAMP_ADD(user_first_engagement, INTERVAL 24 HOUR), 1, 0) AS churned,
  FROM
    (
      SELECT DISTINCT
        user_pseudo_id,
        TIMESTAMP_MICROS(MIN(event_timestamp) OVER(PARTITION BY user_pseudo_id)) AS user_first_engagement,
        TIMESTAMP_MICROS(MAX(event_timestamp) OVER(PARTITION BY user_pseudo_id)) AS user_last_engagement,
        TIMESTAMP_MICROS(MAX(event_timestamp) OVER(PARTITION BY 1)) AS max_ts_events,
      FROM
        `firebase-public-project.analytics_153293282.events_*`
      WHERE
        event_name="user_engagement"
    )
),
user_demographics AS 
  (
    SELECT
      user_pseudo_id,
      country_name,
      device_os,
      device_lang
    FROM 
      (
        SELECT
          user_pseudo_id,
          geo.country as country_name,
          device.operating_system as device_os,
          device.language as device_lang,
          ROW_NUMBER() OVER (PARTITION BY user_pseudo_id ORDER BY event_timestamp DESC) AS row_num
        FROM `firebase-public-project.analytics_153293282.events_*`
        WHERE event_name="user_engagement"
      )
    WHERE row_num = 1
  ),
user_behavior_agg AS 
  (
    SELECT
      user_pseudo_id,
      SUM(IF(event_name = 'user_engagement',          1, 0)) AS cnt_user_engagement,
      SUM(IF(event_name = 'level_start_quickplay',    1, 0)) AS cnt_level_start_quickplay,
      SUM(IF(event_name = 'level_end_quickplay',      1, 0)) AS cnt_level_end_quickplay,
      SUM(IF(event_name = 'level_complete_quickplay', 1, 0)) AS cnt_level_complete_quickplay,
      SUM(IF(event_name = 'level_reset_quickplay',    1, 0)) AS cnt_level_reset_quickplay,
      SUM(IF(event_name = 'post_score',               1, 0)) AS cnt_post_score,
      SUM(IF(event_name = 'spend_virtual_currency',   1, 0)) AS cnt_spend_virtual_currency,
      SUM(IF(event_name = 'ad_reward',                1, 0)) AS cnt_ad_reward,
      SUM(IF(event_name = 'challenge_a_friend',       1, 0)) AS cnt_challenge_a_friend,
      SUM(IF(event_name = 'completed_5_levels',       1, 0)) AS cnt_completed_5_levels,
      SUM(IF(event_name = 'use_extra_steps',          1, 0)) AS cnt_use_extra_steps,
    FROM
      (
        SELECT
          e.*
        FROM
          `firebase-public-project.analytics_153293282.events_*` e
        JOIN
          user_returning r
        ON
          e.user_pseudo_id = r.user_pseudo_id
        WHERE
          TIMESTAMP_MICROS(e.event_timestamp) <= r.ts_after_first_engagement
      )
    GROUP BY
      1
  )
SELECT
  ur.user_first_engagement,
  ur.user_pseudo_id,
  CASE(ABS(MOD(FARM_FINGERPRINT(CAST(ur.user_pseudo_id as STRING)), 10)))
    WHEN 9 THEN 'testing'
    WHEN 8 THEN 'validation'
    ELSE 'training' 
  END AS split_group,
  ur.is_enable,
  ur.bounced,
  ud.country_name,
  ud.device_os,
  ud.device_lang,
  IFNULL(ub.cnt_user_engagement,          0) AS cnt_user_engagement,
  IFNULL(ub.cnt_level_start_quickplay,    0) AS cnt_level_start_quickplay,
  IFNULL(ub.cnt_level_end_quickplay,      0) AS cnt_level_end_quickplay,
  IFNULL(ub.cnt_level_complete_quickplay, 0) AS cnt_level_complete_quickplay,
  IFNULL(ub.cnt_level_reset_quickplay,    0) AS cnt_level_reset_quickplay,
  IFNULL(ub.cnt_post_score,               0) AS cnt_post_score,
  IFNULL(ub.cnt_spend_virtual_currency,   0) AS cnt_spend_virtual_currency,
  IFNULL(ub.cnt_ad_reward,                0) AS cnt_ad_reward,
  IFNULL(ub.cnt_challenge_a_friend,       0) AS cnt_challenge_a_friend,
  IFNULL(ub.cnt_completed_5_levels,       0) AS cnt_completed_5_levels,
  IFNULL(ub.cnt_use_extra_steps,          0) AS cnt_use_extra_steps,
  ur.churned
FROM
  user_returning as ur
LEFT OUTER JOIN
  user_demographics as ud
ON 
  ur.user_pseudo_id = ud.user_pseudo_id
LEFT OUTER JOIN 
  user_behavior_agg as ub
ON
  ur.user_pseudo_id = ub.user_pseudo_id