"""
Example integration scripts untuk Face Recognition API
"""

import requests
import json
from typing import List, Dict, Union


class FaceRecognitionClient:
    """Client untuk Face Recognition API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def health_check(self) -> bool:
        """Check apakah API server berjalan"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def register_face(
        self,
        staff_id: str,
        photo_paths: Dict[str, str]
    ) -> Dict:
        """
        Registrasi wajah petugas
        
        Args:
            staff_id: ID unik petugas
            photo_paths: Dict dengan keys 'front', 'left', 'right', 'top'
                         yang berisi path ke file foto
        
        Returns:
            Response dari API
        """
        files = {
            "photo_front": open(photo_paths["front"], "rb"),
            "photo_left": open(photo_paths["left"], "rb"),
            "photo_right": open(photo_paths["right"], "rb"),
            "photo_top": open(photo_paths["top"], "rb"),
        }
        
        params = {"staff_id": staff_id}
        
        try:
            response = requests.post(
                f"{self.base_url}/register",
                params=params,
                files=files,
                timeout=60
            )
            return response.json()
        finally:
            for f in files.values():
                f.close()
    
    def recognize_face(
        self,
        image_url: str,
        water_channel_door_id: str
    ) -> Dict:
        """
        Pengenalan wajah dari image URL
        
        Args:
            image_url: URL dari image
            water_channel_door_id: ID dari water channel door
        
        Returns:
            Response dari API
        """
        params = {
            "image_url": image_url,
            "water_channel_door_id": water_channel_door_id
        }
        
        response = requests.post(
            f"{self.base_url}/recognize",
            params=params,
            timeout=30
        )
        
        return response.json()
    
    def get_registered_staff(self) -> Dict:
        """Dapatkan daftar semua registered staff"""
        response = requests.get(
            f"{self.base_url}/registered-staff",
            timeout=10
        )
        return response.json()


# ============================================================================
# Example Usage
# ============================================================================

def example_1_basic_usage():
    """Example 1: Penggunaan dasar API"""
    print("\n" + "="*60)
    print("Example 1: Basic Usage")
    print("="*60 + "\n")
    
    client = FaceRecognitionClient()
    
    # Check health
    if client.health_check():
        print("✓ API server is running")
    else:
        print("✗ API server is not running")
        return
    
    # Get registered staff
    staff = client.get_registered_staff()
    print(f"\nRegistered Staff: {staff['staff_ids']}")
    print(f"Total: {staff['total']}")


def example_2_recognize_from_url():
    """Example 2: Pengenalan wajah dari image URL"""
    print("\n" + "="*60)
    print("Example 2: Recognize Face from URL")
    print("="*60 + "\n")
    
    client = FaceRecognitionClient()
    
    # Contoh image URL (ganti dengan URL yang sesuai)
    image_url = "https://example.com/face.jpg"
    door_id = "DOOR_ENTRANCE_01"
    
    print(f"Image URL: {image_url}")
    print(f"Door ID: {door_id}")
    
    result = client.recognize_face(image_url, door_id)
    
    print("\nResponse:")
    print(json.dumps(result, indent=2))
    
    # Process hasil
    recognized_ids = result["recognized_ids"]
    
    if recognized_ids == "0":
        print("\n→ No faces detected")
    elif recognized_ids == "unknown":
        print("\n→ Faces detected but not recognized")
    elif isinstance(recognized_ids, list):
        print(f"\n→ Recognized {len(recognized_ids)} staff members:")
        for staff_id, score in zip(recognized_ids, result.get("confidence_scores", [])):
            print(f"  - {staff_id}: {score}%")
    else:
        print(f"\n→ Recognized staff: {recognized_ids}")


def example_3_water_door_integration():
    """Example 3: Integration dengan water channel door system"""
    print("\n" + "="*60)
    print("Example 3: Water Channel Door Integration")
    print("="*60 + "\n")
    
    client = FaceRecognitionClient()
    
    # Simulate multiple door requests
    doors = [
        {"id": "DOOR_01", "image_url": "https://example.com/entrance.jpg"},
        {"id": "DOOR_02", "image_url": "https://example.com/exit.jpg"},
        {"id": "DOOR_03", "image_url": "https://example.com/office.jpg"},
    ]
    
    results = []
    
    for door in doors:
        print(f"Processing: {door['id']}...")
        
        result = client.recognize_face(door["image_url"], door["id"])
        results.append(result)
        
        # Process result
        action = determine_door_action(result)
        print(f"  Action: {action}\n")
    
    return results


def example_4_batch_recognition():
    """Example 4: Batch recognition untuk multiple images"""
    print("\n" + "="*60)
    print("Example 4: Batch Recognition")
    print("="*60 + "\n")
    
    client = FaceRecognitionClient()
    
    # List of images to recognize
    images = [
        {"url": "https://example.com/image1.jpg", "door_id": "DOOR_01"},
        {"url": "https://example.com/image2.jpg", "door_id": "DOOR_02"},
        {"url": "https://example.com/image3.jpg", "door_id": "DOOR_03"},
    ]
    
    results = []
    
    for idx, img in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] Recognizing: {img['url']}")
        
        result = client.recognize_face(img["url"], img["door_id"])
        results.append(result)
        
        print(f"  Result: {result['recognized_ids']}\n")
    
    return results


def determine_door_action(recognition_result: Dict) -> str:
    """
    Tentukan action berdasarkan recognition result
    
    Returns:
        - "OPEN": Buka pintu
        - "DENY": Tolak akses
        - "UNKNOWN": Perlu verifikasi lebih lanjut
    """
    recognized_ids = recognition_result.get("recognized_ids")
    
    if recognized_ids == "0":
        return "UNKNOWN - No faces detected"
    elif recognized_ids == "unknown":
        return "DENY - Unknown person"
    else:
        # Staff recognized
        confidence = recognition_result.get("confidence_scores", [0])[0]
        if confidence > 90:
            return f"OPEN - High confidence ({confidence}%)"
        else:
            return f"UNKNOWN - Low confidence ({confidence}%)"


# ============================================================================
# Webhook Integration Example
# ============================================================================

def example_webhook_handler(request_data: Dict):
    """
    Example webhook handler untuk menerima notification dari camera system
    """
    print("\n" + "="*60)
    print("Example: Webhook Handler")
    print("="*60 + "\n")
    
    # Extract data dari webhook request
    image_url = request_data.get("image_url")
    door_id = request_data.get("door_id")
    timestamp = request_data.get("timestamp")
    
    print(f"Webhook received:")
    print(f"  Image URL: {image_url}")
    print(f"  Door ID: {door_id}")
    print(f"  Timestamp: {timestamp}\n")
    
    # Process dengan Face Recognition API
    client = FaceRecognitionClient()
    result = client.recognize_face(image_url, door_id)
    
    # Determine action
    action = determine_door_action(result)
    
    print(f"Recognition Result: {result['recognized_ids']}")
    print(f"Confidence: {result.get('confidence_scores')}")
    print(f"Action: {action}\n")
    
    # Return response
    return {
        "success": True,
        "door_id": door_id,
        "action": action,
        "recognized_ids": result["recognized_ids"]
    }


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    import sys
    
    examples = {
        "1": ("Basic Usage", example_1_basic_usage),
        "2": ("Recognize from URL", example_2_recognize_from_url),
        "3": ("Water Door Integration", example_3_water_door_integration),
        "4": ("Batch Recognition", example_4_batch_recognition),
    }
    
    print("\n" + "="*60)
    print("Face Recognition API - Examples")
    print("="*60)
    print("\nAvailable Examples:")
    
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    
    print("\nRun examples with: python examples.py <number>")
    print("Example: python examples.py 1\n")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice in examples:
            examples[choice][1]()
        else:
            print(f"Example {choice} not found")
    else:
        # Run all examples
        print("Running all examples...\n")
        for name, func in examples.values():
            try:
                func()
            except Exception as e:
                print(f"Error in {name}: {e}\n")


if __name__ == "__main__":
    main()
