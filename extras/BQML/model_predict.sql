SELECT
    user_pseudo_id,
  , predicted_churned
  , IF(predicted_churned_probs[OFFSET(0)].prob > 0.5, 1, 0)
  , predicted_churned_probs[OFFSET(0)].prob
FROM
  ML.PREDICT(MODEL workshop.churn_xgboost,
  (SELECT * FROM workshop.train))