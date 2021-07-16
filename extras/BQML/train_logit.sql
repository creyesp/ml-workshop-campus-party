CREATE OR REPLACE MODEL workshop.churn_logreg
OPTIONS(
  MODEL_TYPE="LOGISTIC_REG",
  INPUT_LABEL_COLS=["churned"]
  
--   OPTIMIZE_STRATEGY ='BATCH_GRADIENT_DESCENT'  # { 'AUTO_STRATEGY' | 'BATCH_GRADIENT_DESCENT' | 'NORMAL_EQUATION' },
--   L1_REG = 0.1,
--   L2_REG = 0.1,
--   MAX_ITERATIONS = 4,
--   LEARN_RATE_STRATEGY = 'LINE_SEARCH'  # { 'LINE_SEARCH' | 'CONSTANT' },
--   LEARN_RATE = float64_value,
--   EARLY_STOP = { TRUE | FALSE },
--   MIN_REL_PROGRESS = float64_value,
--   DATA_SPLIT_METHOD = { 'AUTO_SPLIT' | 'RANDOM' | 'CUSTOM' | 'SEQ' | 'NO_SPLIT' },
--   DATA_SPLIT_EVAL_FRACTION = float64_value,
--   DATA_SPLIT_COL = string_value,
--   LS_INIT_LEARN_RATE = float64_value,
--   WARM_START = { TRUE | FALSE },
--   AUTO_CLASS_WEIGHTS = { TRUE | FALSE },
--   CLASS_WEIGHTS = struct_array
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
  AND ABS(MOD(FARM_FINGERPRINT(CAST(ur.user_pseudo_id as STRING)), 10)) != 9
