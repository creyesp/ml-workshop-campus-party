SELECT
  *
FROM 
  ML.RECOMMEND(MODEL proyect.rec_model, 
  		(
  			SELECT
  			  user_id
  			FROM 
  			  table
  			WHERE
  			  visitTime > TIME_DIFF(CURRENT_TIME(), 1 HOUR)
		)

  	)
