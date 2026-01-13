import os
import numpy as np
from insightface.app import FaceAnalysis


class FaceRecognition:
    def __init__(self, db_path="face_db", threshold=0.50):
        self.db_path = db_path
        self.threshold = threshold

        self.model = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        self.model.prepare(ctx_id=-1)

        self.embeddings = {}
        self.load_database()

    def load_database(self):
        os.makedirs(self.db_path, exist_ok=True)

        for f in os.listdir(self.db_path):
            if f.endswith(".npy"):
                name = f.replace(".npy", "")
                self.embeddings[name] = np.load(
                    os.path.join(self.db_path, f)
                )

    # ================= REGISTER =================
    def register(self, name, images):
        embs = []
        for img in images:
            faces = self.model.get(img)
            if faces:
                embs.append(faces[0].embedding)

        if not embs:
            return False

        avg_emb = np.mean(embs, axis=0)
        np.save(os.path.join(self.db_path, f"{name}.npy"), avg_emb)
        self.embeddings[name] = avg_emb
        return True

    # ================= IMAGE / FRAME RECOGNITION =================
    def recognize(self, image):
        results = []
        faces = self.model.get(image)

        for face in faces:
            box = face.bbox.astype(int)
            name, score = self.recognize_embedding(face.embedding)
            results.append((box, name, score))

        return results

    # ================= CORE MATCHING =================
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
