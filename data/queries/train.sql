# Train dataset

CREATE OR REPLACE VIEW 
`peya-data-analyt-factory-stg.workshop.train` AS (
  SELECT
    dem.*,
    IFNULL(beh.cnt_user_engagement,          0) AS cnt_user_engagement,
    IFNULL(beh.cnt_level_start_quickplay,    0) AS cnt_level_start_quickplay,
    IFNULL(beh.cnt_level_end_quickplay,      0) AS cnt_level_end_quickplay,
    IFNULL(beh.cnt_level_complete_quickplay, 0) AS cnt_level_complete_quickplay,
    IFNULL(beh.cnt_level_reset_quickplay,    0) AS cnt_level_reset_quickplay,
    IFNULL(beh.cnt_post_score,               0) AS cnt_post_score,
    IFNULL(beh.cnt_spend_virtual_currency,   0) AS cnt_spend_virtual_currency,
    IFNULL(beh.cnt_ad_reward,                0) AS cnt_ad_reward,
    IFNULL(beh.cnt_challenge_a_friend,       0) AS cnt_challenge_a_friend,
    IFNULL(beh.cnt_completed_5_levels,       0) AS cnt_completed_5_levels,
    IFNULL(beh.cnt_use_extra_steps,          0) AS cnt_use_extra_steps,
    ret.user_first_engagement,
    ret.month,
    ret.julianday,
    ret.dayofweek,
    ret.churned
  FROM
    `peya-data-analyt-factory-stg.workshop.returningusers` ret
  LEFT OUTER JOIN
    `peya-data-analyt-factory-stg.workshop.user_demographics` dem
  ON 
    ret.user_pseudo_id = dem.user_pseudo_id
  LEFT OUTER JOIN 
    `peya-data-analyt-factory-stg.workshop.user_aggregate_behavior` beh
  ON
    ret.user_pseudo_id = beh.user_pseudo_id
  WHERE ret.bounced = 0
  )