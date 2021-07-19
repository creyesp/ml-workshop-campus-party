CREATE OR REPLACE MODEL workshop.churn_dnn

TRANSFORM(
--   EXTRACT(MONTH from TIMESTAMP_MILLIS(user_first_engagement)) as month,
--   EXTRACT(DAYOFYEAR from TIMESTAMP_MILLIS(user_first_engagement)) as julianday,
--   EXTRACT(DAYOFWEEK from TIMESTAMP_MILLIS(user_first_engagement)) as dayofweek,
--   EXTRACT(HOUR from TIMESTAMP_MILLIS(user_first_engagement)) as hour,
  * EXCEPT(user_first_engagement, user_pseudo_id)
)
OPTIONS(
  MODEL_TYPE="DNN_CLASSIFIER",
  INPUT_LABEL_COLS=["churned"]
) AS

SELECT 
  * 
FROM
  workshop.flood_it_dataset
WHERE
  is_enable = 1
  AND bounced = 0
  AND ABS(MOD(FARM_FINGERPRINT(CAST(user_pseudo_id as STRING)), 10)) != 9