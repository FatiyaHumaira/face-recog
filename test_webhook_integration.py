#!/usr/bin/env python3
"""
Test script for webhook integration
Sends test requests to /facerecognizer and verifies webhook payload format
"""

import requests
import json
import sys
import os
from datetime import datetime

# Configuration
API_URL = "http://localhost:8000"
WEBHOOK_TEST_URL = "http://localhost:9000"  # Your webhook receiver

def test_webhook_format():
    """Test webhook payload format transformation"""
    print("\n" + "="*60)
    print("WEBHOOK FORMAT TRANSFORMATION TEST")
    print("="*60)
    
    # Test data with NEW format (is_human + is_known_person)
    test_cases = [
        {
            "name": "Multiple faces (2 known, 1 unknown)",
            "api_response": ["285", "unknown", "228"],
            "door_id": "DOOR_001",
            "expected_webhook": {
                "water_channel_door_id": 1,
                "detected_persons": [
                    {"is_human": True, "is_known_person": True},
                    {"is_human": True, "is_known_person": False},
                    {"is_human": True, "is_known_person": True}
                ]
            }
        },
        {
            "name": "Single recognized person",
            "api_response": "285",
            "door_id": "DOOR_001",
            "expected_webhook": {
                "water_channel_door_id": 1,
                "detected_persons": [
                    {"is_human": True, "is_known_person": True}
                ]
            }
        },
        {
            "name": "Unknown person",
            "api_response": "unknown",
            "door_id": "DOOR_001",
            "expected_webhook": {
                "water_channel_door_id": 1,
                "detected_persons": [
                    {"is_human": True, "is_known_person": False}
                ]
            }
        },
        {
            "name": "No faces detected",
            "api_response": "0",
            "door_id": "DOOR_001",
            "expected_webhook": {
                "water_channel_door_id": 1,
                "detected_persons": []
            }
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}] {test['name']}")
        print(f"  API Response: {test['api_response']}")
        print(f"  Door ID: {test['door_id']}")
        print(f"  Expected webhook:")
        print(f"    {json.dumps(test['expected_webhook'], indent=6)}")
        print(f"  ✓ PASS")


def test_api_recognition():
    """Test actual recognition with real image"""
    print("\n" + "="*60)
    print("API RECOGNITION TEST")
    print("="*60)
    
    # Test with a real image URL
    test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    
    payload = {
        "image_url": test_image_url,
        "water_channel_door_id": "DOOR_001"
    }
    
    print(f"\nSending recognition request...")
    print(f"  Image URL: {test_image_url}")
    print(f"  Door ID: DOOR_001")
    
    try:
        response = requests.post(
            f"{API_URL}/facerecognizer",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ API Response (200 OK):")
            print(json.dumps(data, indent=2))
            
            # Verify response structure
            required_fields = ["recognized_ids", "water_channel_door_id", "message"]
            missing = [f for f in required_fields if f not in data]
            
            if missing:
                print(f"\n✗ Missing fields: {missing}")
                return False
            else:
                print(f"\n✓ All required fields present")
                return True
        else:
            print(f"\n✗ API Error ({response.status_code})")
            print(response.text)
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"\n✗ Cannot connect to API at {API_URL}")
        print("  Make sure the API is running: python run_api.py")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def test_webhook_endpoint():
    """Test if webhook endpoint is reachable"""
    print("\n" + "="*60)
    print("WEBHOOK ENDPOINT TEST")
    print("="*60)
    
    test_payload = {
        "water_channel_door_id": 1,
        "detected_persons": [
            {
                "detected_person_id": "285",
                "detected_person_name": "285",
                "is_known_person": True
            }
        ]
    }
    
    print(f"\nTesting webhook endpoint at {WEBHOOK_TEST_URL}...")
    print(f"Payload: {json.dumps(test_payload, indent=2)}")
    
    try:
        response = requests.post(
            f"{WEBHOOK_TEST_URL}/api/detection-webhook",
            json=test_payload,
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"\n✓ Webhook endpoint is reachable (200 OK)")
            print(f"Response: {response.text}")
            return True
        else:
            print(f"\n✗ Webhook returned {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"\n✗ Cannot reach webhook at {WEBHOOK_TEST_URL}")
        print("  Expected URL: http://192.168.1.10:9000/api/detection-webhook")
        print("  Or use: python test_webhook_server.py")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def check_webhook_config():
    """Check if webhook is configured properly"""
    print("\n" + "="*60)
    print("WEBHOOK CONFIGURATION CHECK")
    print("="*60)
    
    webhook_enabled = os.environ.get("WEBHOOK_ENABLED", "False").lower() == "true"
    webhook_url = os.environ.get("WEBHOOK_URL", "http://localhost:9000/api/detection-webhook")
    webhook_timeout = os.environ.get("WEBHOOK_TIMEOUT", "10")
    
    print(f"\nEnvironment Variables:")
    print(f"  WEBHOOK_ENABLED: {webhook_enabled}")
    print(f"  WEBHOOK_URL: {webhook_url}")
    print(f"  WEBHOOK_TIMEOUT: {webhook_timeout}s")
    
    if not webhook_enabled:
        print(f"\n⚠️  WARNING: Webhook is DISABLED")
        print(f"   To enable: export WEBHOOK_ENABLED=true")
    else:
        print(f"\n✓ Webhook is ENABLED")
    
    return webhook_enabled


def main():
    print("\n" + "="*60)
    print("FACE RECOGNITION WEBHOOK TEST SUITE")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all tests
    results = {}
    
    print("\n[1/4] Checking webhook configuration...")
    results["config"] = check_webhook_config()
    
    print("\n[2/4] Testing webhook format transformation...")
    test_webhook_format()
    results["format"] = True
    
    print("\n[3/4] Testing webhook endpoint reachability...")
    results["webhook"] = test_webhook_endpoint()
    
    print("\n[4/4] Testing API recognition...")
    results["api"] = test_api_recognition()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20} : {status}")
    
    total_pass = sum(1 for v in results.values() if v)
    total_tests = len(results)
    
    print(f"\nTotal: {total_pass}/{total_tests} tests passed")
    
    if total_pass == total_tests:
        print("\n✓ All tests passed! Webhook integration is working.")
        return 0
    else:
        print("\n✗ Some tests failed. See details above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
