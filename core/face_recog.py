import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceRecognition:
    def __init__(
        self,
        db_path="face_db",
        threshold=0.45,
        blur_thresh=100.0,
        max_yaw=30
    ):
        self.db_path = db_path
        self.threshold = threshold
        self.blur_thresh = blur_thresh
        self.max_yaw = max_yaw

        self.model = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        self.model.prepare(ctx_id=-1)

        self.embeddings = {}
        self.load_database()

    # Function Database
    def load_database(self):
        os.makedirs(self.db_path, exist_ok=True)

        for f in os.listdir(self.db_path):
            if f.endswith(".npy"):
                name = f.replace(".npy", "")
                self.embeddings[name] = np.load(
                    os.path.join(self.db_path, f)
                )

    # Function cek blur dan side face
    def is_blurry(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < self.blur_thresh

    def is_side_face(self, face):
        yaw = abs(face.pose[0])  
        return yaw > self.max_yaw

    # Function register face
    def register(self, name, frames):
        # Cek apakah namanya udah ada atau belum
        if name in self.embeddings:
            return False, "Name already exists"

        embeddings = []

        for frame in frames:
            faces = self.model.get(frame)
            if len(faces) != 1:
                continue

            emb = faces[0].embedding
            embeddings.append(emb)

        if len(embeddings) < 3:
            return False, "Not enough valid face samples"

        # Rata-rata embedding
        mean_embedding = np.mean(embeddings, axis=0)
        self.embeddings[name] = mean_embedding

        # -Menyimpan embedding ke database dalam format npy
        np.save(os.path.join(self.db_path, f"{name}.npy"), mean_embedding)

        return True, "Registration successful"


    # Fungction recognize (untuk mengambil bounding box dan nilai embedding wajah)
    def recognize(self, image):
        results = []
        faces = self.model.get(image)

        for face in faces:
            box = face.bbox.astype(int)
            name, score = self.recognize_embedding(face.embedding)
            results.append((box, name, score))

        return results

    # Function recognize embedding (untuk mencocokkan embedding wajah dengan database)
    def recognize_embedding(self, emb):
        best_name = "Unknown"
        best_score = 0.0

        for name, db_emb in self.embeddings.items():
            score = np.dot(emb, db_emb) / (
                np.linalg.norm(emb) * np.linalg.norm(db_emb)
            )

            if score > best_score:
                best_score = score
                best_name = name

        if best_score < self.threshold:
            return "Unknown", float(best_score)

        return best_name, float(best_score)
