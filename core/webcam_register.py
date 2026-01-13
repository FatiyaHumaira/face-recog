import cv2
import numpy as np
import os

def register_from_webcam(face_recog, name, save_path, min_samples=5):
    cap = cv2.VideoCapture(0)
    collected = []

    print("[INFO] SPACE = capture | ESC = finish")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_recog.model.get(frame)

        for face in faces:
            x1,y1,x2,y2 = face.bbox.astype(int)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.putText(
            frame,
            f"Captured: {len(collected)}/{min_samples}",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,255),
            2
        )

        cv2.imshow("Register Face", frame)
        key = cv2.waitKey(1)

        if key == 32 and faces:  # SPACE
            collected.append(faces[0].embedding)
            print(f"[CAPTURE] {len(collected)}")

        elif key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(collected) >= min_samples:
        avg = np.mean(collected, axis=0)
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, f"{name}.npy"), avg)
        face_recog.embeddings[name] = avg
        return True

    return False
