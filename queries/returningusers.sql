# for the churned column, churned=0 if the user performs an action after 24 hours since their first touch, otherwise if their last action was only within the first 24 hours, then churned=1.
# For the bounced column, bounced=1 if the user's last action was within the first ten minutes since their first touch with the app, otherwise bounced=0. We can use this column to filter our training data later on, by conditionally querying for users where bounced = 0.
CREATE OR REPLACE VIEW
  `peya-data-analyt-factory-stg.workshop.returningusers` AS (
  WITH
    firstlasttouch AS (
    SELECT
      user_pseudo_id,
      MIN(event_timestamp) AS user_first_engagement,
      MAX(event_timestamp) AS user_last_engagement
    FROM
      `firebase-public-project.analytics_153293282.events_*`
    WHERE
      event_name="user_engagement"
    GROUP BY
      user_pseudo_id )
  SELECT
    user_pseudo_id,
    user_first_engagement,
    user_last_engagement,
    EXTRACT(MONTH FROM TIMESTAMP_MICROS(user_first_engagement)) AS month,
    EXTRACT(DAYOFYEAR FROM TIMESTAMP_MICROS(user_first_engagement)) AS julianday,
    EXTRACT(DAYOFWEEK FROM TIMESTAMP_MICROS(user_first_engagement)) AS dayofweek,
    (user_first_engagement + 86400000000) AS ts_24hr_after_first_engagement,
    IF (user_last_engagement < (user_first_engagement + 86400000000), 1, 0) AS churned,
    IF (user_last_engagement <= (user_first_engagement + 600000000), 1, 0) AS bounced,
  FROM
    firstlasttouch
  GROUP BY
    1, 2, 3
)
