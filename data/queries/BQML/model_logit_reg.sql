CREATE OR REPLACE MODEL `peya-data-analyt-factory-stg.workshop.churn_logreg`

TRANSFORM(
  EXTRACT(MONTH from TIMESTAMP_MILLIS(user_first_engagement)) as month,
  EXTRACT(DAYOFYEAR from TIMESTAMP_MILLIS(user_first_engagement)) as julianday,
  EXTRACT(DAYOFWEEK from TIMESTAMP_MILLIS(user_first_engagement)) as dayofweek,
  EXTRACT(HOUR from TIMESTAMP_MILLIS(user_first_engagement)) as hour,
  * EXCEPT(user_first_engagement, user_pseudo_id)
)
OPTIONS(
  MODEL_TYPE="LOGISTIC_REG",
  INPUT_LABEL_COLS=["churned"]
  
  OPTIMIZE_STRATEGY ='BATCH_GRADIENT_DESCENT'  # { 'AUTO_STRATEGY' | 'BATCH_GRADIENT_DESCENT' | 'NORMAL_EQUATION' },
  L1_REG = 0.1,
  L2_REG = 0.1,
  MAX_ITERATIONS = 4,
  LEARN_RATE_STRATEGY = 'LINE_SEARCH'  # { 'LINE_SEARCH' | 'CONSTANT' },
  LEARN_RATE = float64_value,
  EARLY_STOP = { TRUE | FALSE },
  MIN_REL_PROGRESS = float64_value,
  DATA_SPLIT_METHOD = { 'AUTO_SPLIT' | 'RANDOM' | 'CUSTOM' | 'SEQ' | 'NO_SPLIT' },
  DATA_SPLIT_EVAL_FRACTION = float64_value,
  DATA_SPLIT_COL = string_value,
  LS_INIT_LEARN_RATE = float64_value,
  WARM_START = { TRUE | FALSE },
  AUTO_CLASS_WEIGHTS = { TRUE | FALSE },
  CLASS_WEIGHTS = struct_array
) AS

SELECT 
  * 
FROM 
  `peya-data-analyt-factory-stg.workshop.train`