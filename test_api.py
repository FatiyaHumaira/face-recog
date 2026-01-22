"""
Test script untuk Face Recognition API
Jalankan dengan: python test_api.py
"""

import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"

# Color codes untuk terminal
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RESET = '\033[0m'


def print_section(title):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{title:^60}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")


def print_success(msg):
    print(f"{GREEN}✓ {msg}{RESET}")


def print_error(msg):
    print(f"{RED}✗ {msg}{RESET}")


def print_info(msg):
    print(f"{YELLOW}ℹ {msg}{RESET}")


def test_health():
    """Test health check endpoint"""
    print_section("Testing Health Check Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        
        if response.status_code == 200:
            data = response.json()
            print_success("Health check passed")
            print(f"Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print_error(f"Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Connection error: {e}")
        return False


def test_get_registered_staff():
    """Test get registered staff endpoint"""
    print_section("Testing Get Registered Staff Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}/registered-staff")
        
        if response.status_code == 200:
            data = response.json()
            print_success("Successfully fetched registered staff")
            print(f"Total staff: {data.get('total')}")
            print(f"Staff IDs: {data.get('staff_ids')}")
            return True
        else:
            print_error(f"Failed with status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False


def test_register_face(staff_id: str, photo_paths: dict):
    """
    Test register face endpoint
    
    Args:
        staff_id: ID petugas
        photo_paths: Dict dengan keys 'front', 'left', 'right', 'top' yang berisi path ke file foto
    
    Returns:
        bool: True jika berhasil
    """
    print_section(f"Testing Register Face Endpoint - Staff ID: {staff_id}")
    
    # Cek apakah semua file ada
    for position, path in photo_paths.items():
        if not Path(path).exists():
            print_error(f"Photo file tidak ditemukan: {path}")
            return False
    
    try:
        files = {
            "photo_front": open(photo_paths.get("front"), "rb"),
            "photo_left": open(photo_paths.get("left"), "rb"),
            "photo_right": open(photo_paths.get("right"), "rb"),
            "photo_top": open(photo_paths.get("top"), "rb"),
        }
        
        params = {"staff_id": staff_id}
        
        response = requests.post(f"{BASE_URL}/register", params=params, files=files)
        
        # Close files
        for f in files.values():
            f.close()
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Registration successful for {staff_id}")
            print(f"Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print_error(f"Registration failed with status {response.status_code}")
            print(f"Error: {response.json()}")
            return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False


def test_recognize_face(image_url: str, door_id: str):
    """
    Test recognize face endpoint
    
    Args:
        image_url: URL dari image yang akan dikenali
        door_id: Water channel door ID
    
    Returns:
        bool: True jika berhasil
    """
    print_section(f"Testing Recognize Face Endpoint - Door ID: {door_id}")
    print_info(f"Using image URL: {image_url}")
    
    try:
        params = {
            "image_url": image_url,
            "water_channel_door_id": door_id
        }
        
        response = requests.post(f"{BASE_URL}/recognize", params=params)
        
        if response.status_code == 200:
            data = response.json()
            print_success("Recognition completed")
            print(f"Response: {json.dumps(data, indent=2)}")
            
            recognized_ids = data.get("recognized_ids")
            if recognized_ids == "0":
                print_info("Result: No faces detected")
            elif recognized_ids == "unknown":
                print_info("Result: Faces detected but not recognized")
            else:
                print_info(f"Result: Recognized staff IDs: {recognized_ids}")
            
            return True
        else:
            print_error(f"Recognition failed with status {response.status_code}")
            print(f"Error: {response.json()}")
            return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False


def main():
    print(f"\n{BLUE}{'*'*60}{RESET}")
    print(f"{BLUE}{'Face Recognition API Test Suite':^60}{RESET}")
    print(f"{BLUE}{'*'*60}{RESET}")
    
    # Test 1: Health Check
    if not test_health():
        print_error("\n✗ API Server is not running!")
        print_info("Please start the server with: python -m uvicorn api.app:app --reload")
        return
    
    # Test 2: Get Registered Staff
    test_get_registered_staff()
    
    # Test 3: Register Face (optional, uncomment jika ingin test)
    # Ganti path dan staff_id sesuai kebutuhan
    # test_register_face(
    #     staff_id="TEST_STAFF_001",
    #     photo_paths={
    #         "front": "path/to/front.jpg",
    #         "left": "path/to/left.jpg",
    #         "right": "path/to/right.jpg",
    #         "top": "path/to/top.jpg",
    #     }
    # )
    
    # Test 4: Recognize Face (optional, memerlukan image URL yang valid)
    # test_recognize_face(
    #     image_url="https://example.com/image.jpg",
    #     door_id="DOOR_01"
    # )
    
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{GREEN}{'Test Suite Completed':^60}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")


if __name__ == "__main__":
    main()
