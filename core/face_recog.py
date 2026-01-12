import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceRecognition:
    def __init__(self, db_path="face_db"):
        self.model = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        self.model.prepare(ctx_id=-1)  # CPU
        self.db_path = db_path
        os.makedirs(self.db_path, exist_ok=True)
        self.embeddings = {}
        self.load_database()

    def load_database(self):
        for fname in os.listdir(self.db_path):
            if fname.endswith(".npy"):
                name = fname.replace(".npy", "")
                self.embeddings[name] = np.load(os.path.join(self.db_path, fname))

    def register_face(self, name, image):
        faces = self.model.get(image)
        if faces:
            emb = faces[0].embedding
            np.save(os.path.join(self.db_path, f"{name}.npy"), emb)
            self.embeddings[name] = emb
            return True
        return False

    def recognize(self, image, threshold=0.6):
        faces = self.model.get(image)
        results = []
        for face in faces:
            emb = face.embedding
            name = "Unknown"
            best_sim = 0  # default similarity
            for db_name, db_emb in self.embeddings.items():
                sim = np.dot(emb, db_emb)/(np.linalg.norm(emb)*np.linalg.norm(db_emb))
                if sim > best_sim and sim > threshold:
                    best_sim = sim
                    name = db_name
            results.append((face.bbox.astype(int), name, best_sim))  # tambahkan best_sim
        return results


