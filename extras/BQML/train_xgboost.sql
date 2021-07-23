CREATE OR REPLACE MODEL workshop.churn_xgboost
TRANSFORM(
--   EXTRACT(MONTH from TIMESTAMP_MILLIS(user_first_engagement)) as month,
--   EXTRACT(DAYOFYEAR from TIMESTAMP_MILLIS(user_first_engagement)) as julianday,
--   EXTRACT(DAYOFWEEK from TIMESTAMP_MILLIS(user_first_engagement)) as dayofweek,
--   EXTRACT(HOUR from TIMESTAMP_MILLIS(user_first_engagement)) as hour,
  ML.BUCKETIZE(IFNULL(cnt_user_engagement,          0), [0,1,2,3,4]) AS cnt_user_engagement,
  ML.BUCKETIZE(IFNULL(cnt_level_start_quickplay,    0), [0,1,2,3,4]) AS cnt_level_start_quickplay,
  ML.BUCKETIZE(IFNULL(cnt_level_end_quickplay,      0), [0,1,2,3,4]) AS cnt_level_end_quickplay,
  ML.BUCKETIZE(IFNULL(cnt_level_complete_quickplay, 0), [0,1,2,3,4]) AS cnt_level_complete_quickplay,
  ML.BUCKETIZE(IFNULL(cnt_level_reset_quickplay,    0), [0,1,2,3,4]) AS cnt_level_reset_quickplay,
  ML.BUCKETIZE(IFNULL(cnt_post_score,               0), [0,1,2,3,4]) AS cnt_post_score,
  ML.BUCKETIZE(IFNULL(cnt_spend_virtual_currency,   0), [0,1,2,3,4]) AS cnt_spend_virtual_currency,
  ML.BUCKETIZE(IFNULL(cnt_ad_reward,                0), [0,1,2,3,4]) AS cnt_ad_reward,
  ML.BUCKETIZE(IFNULL(cnt_challenge_a_friend,       0), [0,1,2,3,4]) AS cnt_challenge_a_friend,
  ML.BUCKETIZE(IFNULL(cnt_completed_5_levels,       0), [0,1,2,3,4]) AS cnt_completed_5_levels,
  ML.BUCKETIZE(IFNULL(cnt_use_extra_steps,          0), [0,1,2,3,4]) AS cnt_use_extra_steps,
  churned
)
OPTIONS(
  MODEL_TYPE="BOOSTED_TREE_CLASSIFIER",
  AUTO_CLASS_WEIGHTS = TRUE,
  INPUT_LABEL_COLS=["churned"]
) AS

SELECT
  country_name,
  device_os ,
  device_lang ,
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
  workshop.flood_it_dataset
WHERE
  is_enable = 1
  AND bounced = 0
  AND ABS(MOD(FARM_FINGERPRINT(CAST(user_pseudo_id as STRING)), 10)) != 9