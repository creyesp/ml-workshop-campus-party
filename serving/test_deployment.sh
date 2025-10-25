#!/bin/bash

# Quick Test Script for MLServer Deployment
# This script tests the MLServer endpoint with sample data

echo "🧪 Testing MLServer Deployment"
echo "==============================="

# Check if server is running
echo "1️⃣ Checking server health..."
if curl -s -f http://localhost:8080/v2/models/flood-it-churned > /dev/null; then
    echo "✅ Server is responding"
else
    echo "❌ Server is not responding. Please start MLServer first:"
    echo "   cd serving/"
    echo "   ./start_server.sh"
    exit 1
fi

# Test prediction endpoint
echo ""
echo "2️⃣ Testing prediction endpoint..."

# Sample JSON data
SAMPLE_DATA='{
  "inputs": [{
    "name": "json_data",
    "shape": [1],
    "datatype": "BYTES",
    "data": [{
      "user_first_engagement": "2018-07-01 23:26:11.222000+00:00",
      "user_pseudo_id": "TEST_USER_123",
      "is_enable": 1,
      "bounced": 0,
      "country_name": "United States",
      "device_os": "IOS",
      "device_lang": "en",
      "cnt_user_engagement": 5,
      "cnt_level_start_quickplay": 2,
      "cnt_level_end_quickplay": 1,
      "cnt_level_complete_quickplay": 1,
      "cnt_level_reset_quickplay": 0,
      "cnt_post_score": 2,
      "cnt_spend_virtual_currency": 0,
      "cnt_ad_reward": 1,
      "cnt_challenge_a_friend": 0,
      "cnt_completed_5_levels": 0,
      "cnt_use_extra_steps": 0
    }]
  }]
}'

# Send prediction request
RESPONSE=$(curl -s -X POST http://localhost:8080/v2/models/flood-it-churned/infer \
  -H 'Content-Type: application/json' \
  -d "$SAMPLE_DATA")

# Check if request was successful
if echo "$RESPONSE" | grep -q '"outputs"'; then
    echo "✅ Prediction request successful!"
    echo ""
    echo "📊 Response:"
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
    echo ""
    echo "🎉 MLServer deployment is working correctly!"
    echo ""
    echo "💡 Next steps:"
    echo "   • Integrate with your application using the API endpoints"
    echo "   • Check the README.md for client code examples"
    echo "   • Monitor server logs in mlserver.log"
else
    echo "❌ Prediction request failed!"
    echo "Response: $RESPONSE"
    exit 1
fi