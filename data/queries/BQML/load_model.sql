CREATE OR REPLACE MODEL `peya-data-analyt-factory-stg.workshop.name`
OPTIONS(
  MODEL_TYPE="tensorflow",
  model_path='gs://.../*'
) AS
