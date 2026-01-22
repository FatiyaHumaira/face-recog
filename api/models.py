from pydantic import BaseModel, Field
from typing import List, Optional, Union


class RegisterRequest(BaseModel):
    """Model untuk register endpoint request"""
    staff_id: str = Field(..., description="ID unik petugas")
    # Photos will be handled as FormData in the endpoint


class RecognitionRequest(BaseModel):
    """Model untuk recognize endpoint request (body)"""
    image_url: str = Field(..., description="URL dari image yang akan dikenali")
    water_channel_door_id: str = Field(..., description="ID dari water channel door")
    
    class Config:
        example = {
            "image_url": "https://example.com/camera_door_001.jpg",
            "water_channel_door_id": "DOOR_001"
        }


class RegisterResponse(BaseModel):
    """Model untuk register endpoint response"""
    success: bool = Field(..., description="Status registrasi berhasil/gagal")
    staff_id: str = Field(..., description="ID petugas yang terdaftar")
    message: str = Field(..., description="Pesan status registrasi")
    
    class Config:
        example = {
            "success": True,
            "staff_id": "EMP_001",
            "message": "Registration successful"
        }


class RecognitionResponse(BaseModel):
    """Model untuk recognize endpoint response"""
    recognized_ids: Union[List[str], str] = Field(
        ..., 
        description="Array dari recognized staff IDs, atau 'unknown', atau '0' jika tidak ada face"
    )
    water_channel_door_id: str = Field(
        ...,
        description="ID dari water channel door yang melakukan request"
    )
    message: str = Field(..., description="Pesan status pengenalan")
    
    class Config:
        example_single = {
            "recognized_ids": "EMP_001",
            "water_channel_door_id": "DOOR_001",
            "message": "Face recognized"
        }
        
        example_multiple = {
            "recognized_ids": ["EMP_001", "unknown", "EMP_045"],
            "water_channel_door_id": "DOOR_001",
            "message": "3 faces detected (2 recognized, 1 unknown)"
        }
        
        example_none = {
            "recognized_ids": "0",
            "water_channel_door_id": "DOOR_001",
            "message": "No faces detected"
        }


class DetectedPerson(BaseModel):
    """Model untuk detected person di webhook"""
    is_human: bool = Field(default=True, description="True jika terdeteksi wajah manusia")
    is_known_person: bool = Field(..., description="True jika recognized, False jika unknown")
    
    class Config:
        example_known = {
            "is_human": True,
            "is_known_person": True
        }
        
        example_unknown = {
            "is_human": True,
            "is_known_person": False
        }


class DetectionWebhookReq(BaseModel):
    """Model untuk webhook request ke downstream endpoint"""
    water_channel_door_id: int = Field(..., description="ID dari water channel door (numeric)")
    detected_persons: list = Field(..., description="Array detected persons")
    
    class Config:
        example = {
            "water_channel_door_id": 1,
            "detected_persons": [
                {
                    "is_human": True,
                    "is_known_person": True
                },
                {
                    "is_human": True,
                    "is_known_person": False
                }
            ]
        }

class HealthResponse(BaseModel):
    """Model untuk health check endpoint response"""
    status: str = Field(..., description="Status server")
    message: str = Field(..., description="Pesan status")
    
    class Config:
        example = {
            "status": "healthy",
            "message": "Face Recognition API is running"
        }

