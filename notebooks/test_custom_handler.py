#!/usr/bin/env python3
"""
Test script for custom MLServer handler with JSON input.

This demonstrates how clients can send raw JSON data to the server
without needing pandas or any Python-specific preprocessing.
"""

import requests
import json


def test_custom_handler():
    """Test the custom handler with JSON data"""
    
    print("🧪 Testing Custom MLServer Handler")
    print("=" * 40)
    
    # Raw JSON data - this is what any client (Java, C#, JavaScript, etc) would send
    sample_data = [
        {
            "user_first_engagement": "2018-07-01 23:26:11.222000+00:00",
            "user_pseudo_id": "3ED7B13473F22308319388A0CEBEB7B2", 
            "is_enable": 1,
            "bounced": 0,
            "country_name": "United States",
            "device_os": "IOS",
            "device_lang": "en",
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
        },
        {
            "user_first_engagement": "2018-07-07 06:19:07.233000+00:00",
            "user_pseudo_id": "33C04C115CDB49A0A23F4CCCA62BDA40",
            "is_enable": 1,
            "bounced": 0, 
            "country_name": "United States",
            "device_os": "IOS",
            "device_lang": "en-us",
            "cnt_user_engagement": 3,
            "cnt_level_start_quickplay": 1,
            "cnt_level_end_quickplay": 1,
            "cnt_level_complete_quickplay": 1,
            "cnt_level_reset_quickplay": 0,
            "cnt_post_score": 1,
            "cnt_spend_virtual_currency": 0,
            "cnt_ad_reward": 0,
            "cnt_challenge_a_friend": 0,
            "cnt_completed_5_levels": 0,
            "cnt_use_extra_steps": 0
        }
    ]
    
    print(f"📊 Sample data ({len(sample_data)} records):")
    print(json.dumps(sample_data[0], indent=2))
    print("...")
    
    # Create MLServer V2 payload with JSON data
    payload = {
        "inputs": [
            {
                "name": "json_data",
                "shape": [len(sample_data)],
                "datatype": "BYTES",
                "data": sample_data  # Send JSON objects directly
            }
        ]
    }
    
    print(f"\n📤 Sending prediction request to custom handler...")
    
    try:
        response = requests.post(
            'http://localhost:8080/v2/models/flood-it-churned/infer',
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        print(f"📨 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! Custom handler working!")
            
            # Parse predictions  
            if 'outputs' in result:
                for output in result['outputs']:
                    name = output.get('name', 'Unknown')
                    shape = output.get('shape', [])
                    data = output.get('data', [])
                    
                    print(f"\n📋 Output: {name}")
                    print(f"   Shape: {shape}")
                    
                    if name == 'predictions':
                        print("   🎯 Predictions:")
                        for i, pred in enumerate(data):
                            churn_status = "WILL CHURN" if pred == 1 else "WON'T CHURN"
                            print(f"      Sample {i+1}: {pred} ({churn_status})")
                    
                    elif name == 'probabilities' and len(shape) >= 2:
                        print("   📊 Probabilities:")
                        n_samples = shape[0] 
                        n_classes = shape[1]
                        
                        for i in range(n_samples):
                            start_idx = i * n_classes
                            end_idx = start_idx + n_classes
                            probs = data[start_idx:end_idx]
                            
                            no_churn_prob = probs[0] * 100
                            churn_prob = probs[1] * 100
                            print(f"      Sample {i+1}: No Churn: {no_churn_prob:.1f}%, Churn: {churn_prob:.1f}%")
            
            print(f"\n🎉 Test completed successfully!")
            print(f"\n💡 Benefits of this approach:")
            print(f"   ✓ Clients can send raw JSON (language-agnostic)")
            print(f"   ✓ No pandas/sklearn required on client side")
            print(f"   ✓ Server handles all data preprocessing") 
            print(f"   ✓ Maintains sklearn ColumnTransformer functionality")
            
            return True
            
        else:
            print("❌ Request failed!")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False


def demo_curl_request():
    """Show equivalent curl command for non-Python clients"""
    
    sample_json = {
        "user_first_engagement": "2018-07-01 23:26:11.222000+00:00",
        "user_pseudo_id": "ABC123",
        "is_enable": 1,
        "bounced": 0,
        "country_name": "United States", 
        "device_os": "ANDROID",
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
    }
    
    payload = {
        "inputs": [
            {
                "name": "json_data",
                "shape": [1],
                "datatype": "BYTES", 
                "data": [sample_json]
            }
        ]
    }
    
    print(f"\n🌐 Example curl command for other languages:")
    print(f"=" * 50)
    print(f"curl -X POST http://localhost:8080/v2/models/flood-it-churned/infer \\")
    print(f"  -H 'Content-Type: application/json' \\")
    print(f"  -d '{json.dumps(payload, indent=2)}'")


if __name__ == "__main__":
    success = test_custom_handler()
    
    if success:
        demo_curl_request()
    
    exit(0 if success else 1)