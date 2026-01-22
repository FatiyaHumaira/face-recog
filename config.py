import os
from typing import Optional


class Config:
    """
    Base configuration untuk Face Recognition API
    
    Attributes:
        Face Recognition Settings:
            FACE_DB_PATH: Folder untuk menyimpan .npy embeddings
            RECOGNITION_THRESHOLD: Similarity threshold (0-1)
            BLUR_THRESHOLD: Threshold untuk blur detection
            MAX_YAW: Maximum head rotation angle (degrees)
        
        API Settings:
            API_HOST: Host binding address
            API_PORT: Port number
            API_RELOAD: Auto-reload pada code change
        
        CORS Settings:
            CORS_ORIGINS: Allowed origins untuk cross-origin requests
        
        Model Settings:
            MODEL_NAME: InsightFace model name
            PROVIDERS: Execution providers (CPU/GPU)
    """
    
    # FACE RECOGNITION SETTINGS
    FACE_DB_PATH: str = os.environ.get("FACE_DB_PATH", "face_db")
    RECOGNITION_THRESHOLD: float = float(os.environ.get("RECOGNITION_THRESHOLD", "0.45"))
    BLUR_THRESHOLD: float = float(os.environ.get("BLUR_THRESHOLD", "100.0"))
    MAX_YAW: int = int(os.environ.get("MAX_YAW", "30"))
    
    # API SETTINGS 
    API_HOST: str = os.environ.get("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.environ.get("API_PORT", "8000"))
    API_RELOAD: bool = os.environ.get("API_RELOAD", "True").lower() == "true"
    
    # CORS SETTINGS
    CORS_ORIGINS: list = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:8000",
        "*"  # Allow all origins for development
    ]
    
    # MODEL SETTINGS 
    MODEL_NAME: str = "buffalo_l"  # InsightFace model name
    PROVIDERS: list = ["CPUExecutionProvider"]  # Can also use "CUDAExecutionProvider" for GPU
    
    # IMAGE PROCESSING SETTINGS
    MAX_IMAGE_SIZE: int = 50 * 1024 * 1024  # 50 MB
    SUPPORTED_FORMATS: list = ["jpg", "jpeg", "png", "bmp"]
    
    # DATABASE SETTINGS
    DB_HOST: str = os.environ.get("DB_HOST", "localhost")
    DB_PORT: int = int(os.environ.get("DB_PORT", "5432"))
    DB_NAME: str = os.environ.get("DB_NAME", "face_recognition")
    DB_USER: str = os.environ.get("DB_USER", "face_user")
    DB_PASSWORD: str = os.environ.get("DB_PASSWORD", "secure_password_123")
    
    # WEBHOOK SETTINGS 
    WEBHOOK_ENABLED: bool = os.environ.get("WEBHOOK_ENABLED", "False").lower() == "true"
    WEBHOOK_URL: str = os.environ.get("WEBHOOK_URL", "http://localhost:9000/api/detection-webhook")
    WEBHOOK_TIMEOUT: int = int(os.environ.get("WEBHOOK_TIMEOUT", "10"))  # seconds
    
    # LOGGING
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")


class DevelopmentConfig(Config):
    """Development configuration - dengan debugging enabled"""
    DEBUG: bool = True
    API_RELOAD: bool = True
    LOG_LEVEL: str = "DEBUG"


class ProductionConfig(Config):
    """Production configuration - optimized untuk performance dan security"""
    DEBUG: bool = False
    API_RELOAD: bool = False
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: list = []  # Override untuk restrict origins

    CORS_ORIGINS: list = [
        # Add your production domains here
        "https://yourdomain.com",
    ]


# Get current environment
ENV = os.environ.get("ENVIRONMENT", "development").lower()

if ENV == "production":
    current_config = ProductionConfig()
else:
    current_config = DevelopmentConfig()
