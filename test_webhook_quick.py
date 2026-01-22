"""
Quick test script untuk webhook integration
Test langsung tanpa perlu setup server eksternal
"""

import requests
import json
import time

API_URL = "http://localhost:8000/facerecognizer"

# Test cases
test_cases = [
    {
        "name": "gaon",
        "payload": {
            "image_url": "https://i.pinimg.com/736x/b6/23/88/b623889c8cbea966033404869a974bac.jpg",
            "water_channel_door_id": "278"
        },
        "expected": "recognized staff"
    },
    {
        "name": "Unknown Person",
        "payload": {
            "image_url": "https://images.pexels.com/photos/614810/pexels-photo-614810.jpeg?cs=srgb&dl=pexels-simon-robben-55958-614810.jpg&fm=jpg",
            "water_channel_door_id": "442"
        },
        "expected": "unknown person or no face"
    },
    {
        "name": "No Photo URL",
        "payload": {
            "image_url": "https://invalid-url-404.com/nophoto.jpg",
            "water_channel_door_id": "123"
        },
        "expected": "download error"
    }
]

def test_webhook():
    """Test webhook dengan berbagai scenarios"""
    
    print("\n" + "="*80)
    print("WEBHOOK INTEGRATION TEST")
    print("="*80)
    
    # Check if API is running
    try:
        health = requests.get("http://localhost:8000/health", timeout=2)
        if health.status_code == 200:
            print("✓ API is running")
        else:
            print("✗ API health check failed")
            return
    except:
        print("✗ API is not running. Start it first: python run_api.py")
        return
    
    print("\n📝 Running test cases...\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['name']}")
        print(f"  Door ID: {test['payload']['water_channel_door_id']}")
        print(f"  Expected: {test['expected']}")
        
        try:
            response = requests.post(
                API_URL,
                json=test['payload'],
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"  ✓ Status: 200 OK")
                print(f"  Result: {result.get('recognized_ids', 'N/A')}")
                print(f"  Message: {result.get('message', 'N/A')}")
            else:
                print(f"  ✗ Status: {response.status_code}")
                print(f"  Error: {response.text[:100]}")
        
        except requests.exceptions.Timeout:
            print(f"  ✗ Timeout (API too slow)")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        print()
        time.sleep(1)  # Delay between requests
    
    print("="*80)
    print("\n💡 Check webhook receiver logs to verify payloads were sent")
    print("   - Mock server: Terminal where test_webhook_server.py is running")
    print("   - webhook.site: Check browser dashboard")
    print()


if __name__ == "__main__":
    print("\n🔧 Webhook Integration Tester")
    print("="*80)
    print("\nPrerequisites:")
    print("  1. API server running (python run_api.py)")
    print("  2. WEBHOOK_ENABLED=true in environment")
    print("  3. WEBHOOK_URL set to mock server or webhook.site")
    print()
    
    input("Press Enter to start test...")
    
    test_webhook()
