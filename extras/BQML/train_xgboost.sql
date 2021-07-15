CREATE OR REPLACE MODEL workshop.churn_xgboost
TRANSFORM(
--   EXTRACT(MONTH from TIMESTAMP_MILLIS(user_first_engagement)) as month,
--   EXTRACT(DAYOFYEAR from TIMESTAMP_MILLIS(user_first_engagement)) as julianday,
--   EXTRACT(DAYOFWEEK from TIMESTAMP_MILLIS(user_first_engagement)) as dayofweek,
--   EXTRACT(HOUR from TIMESTAMP_MILLIS(user_first_engagement)) as hour,
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