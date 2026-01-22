import cv2
import numpy as np
import asyncio
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import requests
from io import BytesIO
import logging

from core.face_recog import FaceRecognition
from core.db_helper import DatabaseManager
from api.models import (
    RegisterRequest, 
    RegisterResponse, 
    RecognitionRequest,
    RecognitionResponse,
    DetectedPerson,
    DetectionWebhookReq,
    HealthResponse
)
from api.webhook import transform_to_webhook_format, send_webhook

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path untuk database wajah
FACE_DB_PATH = "face_db"

# Global variable untuk face recognition model
face_recog: FaceRecognition = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager untuk initialize dan cleanup resources
    """
    global face_recog
    
    # Startup
    logger.info("Loading Face Recognition Model...")
    face_recog = FaceRecognition(
        db_path=FACE_DB_PATH,
        threshold=0.45
    )
    logger.info("Model loaded successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="Face Recognition API",
    description="API untuk registrasi dan pengenalan wajah dengan sistem water channel door",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint untuk memverifikasi API berjalan
    """
    return HealthResponse(
        status="healthy",
        message="Face Recognition API is running"
    )


@app.post("/faceregister", response_model=RegisterResponse)
async def register_face(
    staff_id: str = Query(..., description="ID petugas yang akan didaftarkan"),
    photo_front: UploadFile = File(..., description="Foto depan wajah"),
    photo_left: UploadFile = File(..., description="Foto sisi kiri wajah"),
    photo_right: UploadFile = File(..., description="Foto sisi kanan wajah"),
    photo_top: UploadFile = File(..., description="Foto atas wajah"),
):
    """
    Endpoint untuk registrasi wajah dengan 4 foto dari sudut berbeda
    
    Parameters:
    - staff_id: ID unik petugas
    - photo_front: File binary foto depan
    - photo_left: File binary foto kiri
    - photo_right: File binary foto kanan
    - photo_top: File binary foto atas
    
    Returns:
    - success: Status registrasi
    - staff_id: ID petugas yang terdaftar
    - message: Pesan status
    """
    try:
        if not staff_id:
            raise HTTPException(status_code=400, detail="ID is required")
        
        # Read uploaded files
        frames = []
        
        for photo_file, position in [
            (photo_front, "front"),
            (photo_left, "left"),
            (photo_right, "right"),
            (photo_top, "top")
        ]:
            if not photo_file:
                raise HTTPException(
                    status_code=400,
                    detail=f"Photo {position} is required"
                )
            
            # Convert uploaded file to image
            content = await photo_file.read()
            image = cv2.imdecode(
                np.frombuffer(content, np.uint8),
                cv2.IMREAD_COLOR
            )
            
            if image is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid image file for photo {position}"
                )
            
            frames.append(image)
            logger.info(f"Processed photo {position} for staff_id: {staff_id}")
        
        # Register face
        success, message = face_recog.register(staff_id, frames)
        
        if not success:
            raise HTTPException(status_code=400, detail=message)
        
        logger.info(f"Successfully registered staff_id: {staff_id}")
        
        return RegisterResponse(
            success=True,
            staff_id=staff_id,
            message=message
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during registration: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Registration failed: {str(e)}"
        )


@app.post("/facerecognizer", response_model=RecognitionResponse)
async def recognize_face(request: RecognitionRequest):
    """
    Endpoint untuk pengenalan wajah dari URL image
    
    Request Body:
    - image_url: URL image yang akan dikenali
    - water_channel_door_id: ID door yang melakukan request
    
    Returns:
    - recognized_ids: Array ID staff yang dikenali atau "unknown" atau "0"
    - water_channel_door_id: ID door yang sama dengan request
    - message: Pesan status
    """
    try:
        image_url = request.image_url
        water_channel_door_id = request.water_channel_door_id
        
        if not image_url or not water_channel_door_id:
            raise HTTPException(
                status_code=400,
                detail="image_url and water_channel_door_id are required"
            )
        
        logger.info(f"Recognizing face from URL: {image_url}")
        
        # Download image dari URL
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download image from URL: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download image: {str(e)}"
            )
        
        # Convert image bytes to OpenCV format
        image = cv2.imdecode(
            np.frombuffer(response.content, np.uint8),
            cv2.IMREAD_COLOR
        )
        
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image format from URL"
            )
        
        # Recognize faces
        results = face_recog.recognize(image)
        
        if len(results) == 0:
            logger.info(f"No faces detected in image from door: {water_channel_door_id}")
            
            # Send webhook for no faces detected
            try:
                webhook_data = transform_to_webhook_format(
                    recognized_ids="0",
                    water_channel_door_id=water_channel_door_id
                )
                asyncio.create_task(send_webhook(webhook_data))
                logger.debug(f"Webhook task created for no faces (door {water_channel_door_id})")
            except Exception as e:
                logger.warning(f"Failed to queue webhook for no faces: {str(e)}")
            
            return RecognitionResponse(
                recognized_ids="0",
                water_channel_door_id=water_channel_door_id,
                message="No faces detected"
            )
        
        # Process recognition results - collect semua face (recognized + unknown)
        recognized_ids = []
        
        for _, name, score in results:
            recognized_ids.append(name)
        
        # Prepare response
        if len(recognized_ids) == 0:
            # Tidak ada faces sama sekali (seharusnya sudah handled di atas)
            result_ids = "0"
            message = "No faces detected"
        elif len(recognized_ids) == 1:
            # Single face (bisa recognized atau unknown)
            result_ids = recognized_ids[0]
            message = f"Recognized: {recognized_ids[0]}"
        else:
            # Multiple faces
            result_ids = recognized_ids
            message = f"Found {len(recognized_ids)} face(s)"
        
        logger.info(f"Recognition result for door {water_channel_door_id}: {result_ids}")
        
        # Send webhook asynchronously (fire-and-forget)
        try:
            webhook_data = transform_to_webhook_format(
                recognized_ids=result_ids,
                water_channel_door_id=water_channel_door_id
            )
            asyncio.create_task(send_webhook(webhook_data))
            logger.debug(f"Webhook task created for door {water_channel_door_id}")
        except Exception as e:
            logger.warning(f"Failed to queue webhook: {str(e)}")
        
        return RecognitionResponse(
            recognized_ids=result_ids,
            water_channel_door_id=water_channel_door_id,
            message=message
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during recognition: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Recognition failed: {str(e)}"
        )


@app.get("/registered-staff")
async def get_registered_staff():
    """
    Endpoint untuk mendapatkan daftar semua staff yang sudah terdaftar
    """
    try:
        staff_list = list(face_recog.embeddings.keys())
        return {
            "total": len(staff_list),
            "staff_ids": staff_list
        }
    except Exception as e:
        logger.error(f"Error fetching registered staff: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch registered staff: {str(e)}"
        )


@app.post("/reload-database")
async def reload_database():
    """
    Reload embeddings database dari disk.
    Gunakan endpoint ini ketika Anda menghapus/menambah file .npy di folder face_db/
    tanpa perlu restart server.
    """
    try:
        face_recog.reload_database()
        staff_list = list(face_recog.embeddings.keys())
        logger.info(f"Database reloaded. Total staff: {len(staff_list)}")
        
        return {
            "success": True,
            "message": "Database reloaded successfully",
            "total_staff": len(staff_list),
            "staff_ids": staff_list
        }
    except Exception as e:
        logger.error(f"Error reloading database: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload database: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
