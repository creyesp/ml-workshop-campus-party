# Select user data only from first 24 hr of using the app
# Other posible events:
#  ad_reward
#  app_clear_data
#  app_exception
#  app_remove
#  app_update
#  challenge_a_friend
#  challenge_accepted
#  completed_5_levels
#  dynamic_link_app_open
#  dynamic_link_first_open
#  error
#  firebase_campaign
#  first_open
#  in_app_purchase
#  level_complete
#  level_complete_quickplay
#  level_end
#  level_end_quickplay
#  level_fail
#  level_fail_quickplay
#  level_reset
#  level_reset_quickplay
#  level_retry
#  level_retry_quickplay
#  level_start
#  level_start_quickplay
#  level_up
#  no_more_extra_steps
#  notification_foreground
#  os_update
#  post_score
#  screen_view
#  select_content
#  session_start
#  spend_virtual_currency
#  use_extra_steps
#  user_engagement

CREATE OR REPLACE VIEW 
`peya-data-analyt-factory-stg.workshop.user_aggregate_behavior` AS (
WITH
  events_first24hr AS (
    SELECT
      e.*
    FROM
      `firebase-public-project.analytics_153293282.events_*` e
    JOIN
      bqmlga4.returningusers r
    ON
      e.user_pseudo_id = r.user_pseudo_id
    WHERE
      e.event_timestamp <= r.ts_24hr_after_first_engagement
    )
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
  events_first24hr
GROUP BY
  1
  )
