import os
import cv2
import numpy as np
import logging
from insightface.app import FaceAnalysis
from core.db_helper import DatabaseManager

logger = logging.getLogger(__name__)


class FaceRecognition:
    """
    Main class untuk face recognition operations
    
    Menggunakan InsightFace model buffalo_l untuk:
    - Face detection
    - Landmark detection
    - Face embedding generation (512D vector)
    """
    
    def __init__(
        self,
        db_path="face_db",
        threshold=0.45,
        blur_thresh=100.0,
        max_yaw=30
    ):
        """
        Initialize Face Recognition
        
        Args:
            db_path: Path ke folder database embeddings (.npy files)
            threshold: Similarity threshold untuk recognition (0-1)
            blur_thresh: Threshold untuk blur detection
            max_yaw: Maximum head rotation angle (degrees)
        """
        self.db_path = db_path
        self.threshold = threshold
        self.blur_thresh = blur_thresh
        self.max_yaw = max_yaw

        # Load InsightFace model
        print("Loading InsightFace model (buffalo_l)...")
        self.model = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        self.model.prepare(ctx_id=-1)
        print("Model loaded successfully")

        # Dictionary untuk simpan embeddings di memory
        self.embeddings = {}
        
        # Database manager
        self.db_manager = DatabaseManager()
        
        # Load embeddings dari database
        self.load_database()

    # DATABASE FUNCTIONS
    
    def load_database(self):
        """
        Load semua embeddings dari database ke memory
        
        Priority:
        1. Try load dari PostgreSQL database
        2. Fallback ke .npy files di folder (untuk development)
        """
        os.makedirs(self.db_path, exist_ok=True)

        # Try load dari PostgreSQL
        try:
            db_embeddings = self.db_manager.load_all_embeddings()
            if db_embeddings:
                self.embeddings = db_embeddings
                logger.info(f"✓ Loaded {len(self.embeddings)} embeddings from PostgreSQL")
                return
        except Exception as e:
            logger.warning(f"Could not load from PostgreSQL: {e}")
        
        # Fallback: Load dari .npy files (development)
        logger.info("Falling back to .npy files...")
        for f in os.listdir(self.db_path):
            if f.endswith(".npy"):
                name = f.replace(".npy", "")
                self.embeddings[name] = np.load(
                    os.path.join(self.db_path, f)
                )
        
        if self.embeddings:
            logger.info(f"✓ Loaded {len(self.embeddings)} embeddings from .npy files")
        else:
            logger.info("No embeddings found")

    def reload_database(self):
        """
        Reload embeddings dari database/disk
        
        Gunakan fungsi ini untuk sync database ketika ada perubahan.
        """
        self.embeddings.clear()
        self.load_database()
        logger.info("Database reloaded")

    # VALIDATION FUNCTIONS 
    
    def is_blurry(self, img):
        """
        Check apakah image terlalu blur (Laplacian variance method)
        
        Args:
            img: Input image (BGR format)
            
        Returns:
            Boolean - True jika blur, False jika clear
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < self.blur_thresh

    def is_side_face(self, face):
        """
        Check apakah face terlalu side/profile
        
        Args:
            face: Face object dari InsightFace
            
        Returns:
            Boolean - True jika side face (yaw > max_yaw), False jika front
        """
        yaw = abs(face.pose[0])  # pose[0] adalah yaw angle
        return yaw > self.max_yaw

    # REGISTRATION FUNCTIONS
    
    def register(self, name, frames):
        """
        Register wajah dari multiple frames (4 photos)
        
        Args:
            name: Unique identifier untuk staff (e.g., 'EMP_001')
            frames: List dari image frames (OpenCV BGR format)
            
        Returns:
            (success: bool, message: str)
        """
        # Cek apakah staff sudah terdaftar
        if name in self.embeddings:
            return False, "Staff already registered"

        embeddings = []

        # Extract embedding dari setiap frame
        for frame in frames:
            faces = self.model.get(frame)
            
            # Hanya ambil jika ada exactly 1 face
            if len(faces) != 1:
                continue

            # Skip jika face blur atau side face
            
            emb = faces[0].embedding
            embeddings.append(emb)

        # Minimum 3 embeddings yang valid
        if len(embeddings) < 3:
            return False, "Not enough valid face samples (minimum 3)"

        # Calculate mean embedding dari 4 photos
        mean_embedding = np.mean(embeddings, axis=0)
        self.embeddings[name] = mean_embedding

        # Save embedding ke database (BYTEA)
        success = self.db_manager.save_embedding(name, mean_embedding)
        
        if not success:
            logger.error(f"Failed to save embedding to database for {name}")
            # Still save to .npy as backup
            np.save(
                os.path.join(self.db_path, f"{name}.npy"),
                mean_embedding
            )
            return True, "Registration successful (saved to backup)"
        
        # Also save to .npy as backup
        np.save(
            os.path.join(self.db_path, f"{name}.npy"),
            mean_embedding
        )

        return True, "Registration successful"

    def register_single_photo(self, image_bytes, name):
        """
        Register dari single photo (digunakan untuk bulk registration)
        
        Args:
            image_bytes: Raw image bytes (dari requests.get().content)
            name: Staff ID untuk disimpan
        
        Returns:
            embedding array (512D) atau None jika gagal
        """
        try:
            # Convert bytes ke OpenCV image
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return None
            
            # Detect faces di image
            faces = self.model.get(img)
            if len(faces) == 0:
                return None
            
            # Return embedding dari face pertama
            return faces[0].embedding
            
        except Exception as e:
            print(f"Error registering {name}: {e}")
            return None

    # RECOGNITION FUNCTIONS
    
    def recognize(self, image):
        """
        Recognize faces di image
        
        Args:
            image: Input image (OpenCV BGR format)
            
        Returns:
            List dari (box, name, score) tuples
            - box: [x1, y1, x2, y2] bounding box
            - name: recognized staff_id atau 'Unknown'
            - score: similarity score (0-1)
        """
        results = []
        faces = self.model.get(image)

        for face in faces:
            box = face.bbox.astype(int)
            name, score = self.recognize_embedding(face.embedding)
            results.append((box, name, score))

        return results

    def recognize_embedding(self, emb):
        """
        Match embedding terhadap semua embeddings di database menggunakan cosine similarity
        
        Args:
            emb: Input embedding (512D vector)
            
        Returns:
            (best_name: str, best_score: float)
            - best_name: recognized staff_id atau 'Unknown'
            - best_score: similarity score (0-1)
        """
        best_name = "unknown"
        best_score = 0.0
        
        # Expected embedding dimension (buffalo_l = 512D)
        expected_dim = emb.shape[0]
        mismatched_count = 0

        # Iterate semua embeddings di database
        for name, db_emb in self.embeddings.items():
            # Validate dimension match
            if db_emb.shape[0] != expected_dim:
                mismatched_count += 1
                logger.warning(
                    f"Dimension mismatch for {name}: "
                    f"expected {expected_dim}D, got {db_emb.shape[0]}D (skipping)"
                )
                continue
            
            # Calculate cosine similarity
            try:
                score = np.dot(emb, db_emb) / (
                    np.linalg.norm(emb) * np.linalg.norm(db_emb)
                )
            except Exception as e:
                logger.error(f"Error calculating similarity for {name}: {e}")
                continue

            # Update jika score lebih tinggi
            if score > best_score:
                best_score = score
                best_name = name
        
        # Log if found mismatched embeddings
        if mismatched_count > 0:
            logger.warning(
                f"Found {mismatched_count} embeddings with wrong dimensions. "
                f"Run cleanup or re-register these staff."
            )

        # Jika score dibawah threshold, return unknown
        if best_score < self.threshold:
            return "unknown", float(best_score)

        return best_name, float(best_score)

