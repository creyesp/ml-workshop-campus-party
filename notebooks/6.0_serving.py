import requests
import pandas as pd
from datetime import datetime

# Example data from comment:
# user_first_engagement,user_pseudo_id,is_enable,bounced,country_name,device_os,device_lang,cnt_user_engagement,cnt_level_start_quickplay,cnt_level_end_quickplay,cnt_level_complete_quickplay,cnt_level_reset_quickplay,cnt_post_score,cnt_spend_virtual_currency,cnt_ad_reward,cnt_challenge_a_friend,cnt_completed_5_levels,cnt_use_extra_steps,churned
# 2018-07-01 23:26:11.222000+00:00,3ED7B13473F22308319388A0CEBEB7B2,1,0,United States,IOS,en,2,0,0,0,0,0,0,0,0,0,0,1

# Create a sample input that matches the expected column order for the model
# The model expects ALL columns (including ignored ones) because preprocessing drops them
sample_data = {
    # IGNORE_COLUMNS (will be dropped by preprocessing)
    "user_first_engagement": "2018-07-01 23:26:11.222000+00:00",
    "user_pseudo_id": "3ED7B13473F22308319388A0CEBEB7B2", 
    "is_enable": 1,
    "bounced": 0,
    "device_lang": "en",  # Note: device_lang appears in both CATEGORICAL and IGNORE
    
    # CATEGORICAL_COLUMNS (processed by categorical transformer)
    "country_name": "United States",
    "device_os": "IOS",
    # "device_lang": "en",  # Already included above
    
    # NUMERICAL_COLUMNS (processed by numeric transformer)
    "cnt_user_engagement": 2,
    "cnt_level_start_quickplay": 0,
    "cnt_level_end_quickplay": 0,
    "cnt_level_complete_quickplay": 0,
    "cnt_level_reset_quickplay": 0,
    "cnt_post_score": 0,
    "cnt_spend_virtual_currency": 0,
    "cnt_ad_reward": 0,
    "cnt_challenge_a_friend": 0,
    "cnt_completed_5_levels": 0,
    "cnt_use_extra_steps": 0
}

# Convert to DataFrame to match the expected input format
df = pd.DataFrame([sample_data])

# Create MLServer payload with the raw data (before preprocessing)
# The model pipeline will handle the preprocessing internally
payload = {
    "inputs": [
        {
            "name": "input_data",
            "datatype": "BYTES",  # Use BYTES for mixed data types (strings and numbers)
            "shape": [1],  # 1 sample
            "data": df.to_json(orient='records'),  # Send as JSON string
            "parameters": {
                "content_type": "pd"  # Indicate this is pandas data
            }
        }
    ]
}

# Alternative: Send as individual columns if the server expects structured input
payload_structured = {
    "inputs": [
        {
            "name": "data",
            "datatype": "BYTES",
            "shape": [len(df.columns)],
            "data": [str(value) for value in df.iloc[0].values],
            "parameters": {
                "columns": df.columns.tolist()
            }
        }
    ]
}

print("Sample data:")
print(df)
print("\nPayload:")
print(payload)

# Make prediction request
response = requests.post(
    "http://localhost:8080/v2/models/flood-it-churned/infer",
    json=payload_structured
)

print("\nResponse:")
print("Status Code:", response.status_code)
if response.status_code == 200:
    result = response.json()
    print("Prediction:", result)
else:
    print("Error:", response.text)