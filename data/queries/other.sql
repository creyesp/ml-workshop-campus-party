SELECT 
  event_params.value.string_value,
  count(1)
FROM 
  `firebase-public-project.analytics_153293282.events_20181003`, unnest(event_params) as event_params
WHERE 
  event_name = 'user_engagement' 
  AND event_params.key = 'firebase_screen_class'
GROUP BY 
  1
ORDER BY 
  2 DESC