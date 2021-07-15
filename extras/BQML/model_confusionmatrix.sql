SELECT
  expected_label,
  _0 AS predicted_0,
  _1 AS predicted_1
FROM
  ML.CONFUSION_MATRIX(MODEL `peya-data-analyt-factory-stg.workshop.churn_xgboost`)