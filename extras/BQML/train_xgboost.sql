CREATE OR REPLACE MODEL workshop.churn_xgboost
TRANSFORM(
--   EXTRACT(MONTH from TIMESTAMP_MILLIS(user_first_engagement)) as month,
--   EXTRACT(DAYOFYEAR from TIMESTAMP_MILLIS(user_first_engagement)) as julianday,
--   EXTRACT(DAYOFWEEK from TIMESTAMP_MILLIS(user_first_engagement)) as dayofweek,
--   EXTRACT(HOUR from TIMESTAMP_MILLIS(user_first_engagement)) as hour,
  IFNULL(cnt_user_engagement,          0) AS cnt_user_engagement,
  IFNULL(cnt_level_start_quickplay,    0) AS cnt_level_start_quickplay,
  IFNULL(cnt_level_end_quickplay,      0) AS cnt_level_end_quickplay,
  IFNULL(cnt_level_complete_quickplay, 0) AS cnt_level_complete_quickplay,
  IFNULL(cnt_level_reset_quickplay,    0) AS cnt_level_reset_quickplay,
  IFNULL(cnt_post_score,               0) AS cnt_post_score,
  IFNULL(cnt_spend_virtual_currency,   0) AS cnt_spend_virtual_currency,
  IFNULL(cnt_ad_reward,                0) AS cnt_ad_reward,
  IFNULL(cnt_challenge_a_friend,       0) AS cnt_challenge_a_friend,
  IFNULL(cnt_completed_5_levels,       0) AS cnt_completed_5_levels,
  IFNULL(cnt_use_extra_steps,          0) AS cnt_use_extra_steps,
  * EXCEPT(user_first_engagement, user_pseudo_id)
)
OPTIONS(
  MODEL_TYPE="BOOSTED_TREE_CLASSIFIER",
  INPUT_LABEL_COLS=["churned"]
) AS

SELECT
  country_name,
  operating_system,
  language,
  cnt_user_engagement,
  cnt_level_start_quickplay,
  cnt_level_end_quickplay,
  cnt_level_complete_quickplay,
  cnt_level_reset_quickplay,
  cnt_post_score,
  cnt_spend_virtual_currency,
  cnt_ad_reward,
  cnt_challenge_a_friend,
  cnt_completed_5_levels,
  cnt_use_extra_steps,
  churned
FROM
  workshop.floot_it_dataset
WHERE
  is_enable = 1
  AND bounced = 0