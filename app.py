import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
from core.face_recog import FaceRecognition

# ================= CONFIG =================
FACE_DB_PATH = "face_db"

@st.cache_resource
def load_model():
    return FaceRecognition(db_path=FACE_DB_PATH, threshold=0.50)

face_recog = load_model()

# ================= HELPER =================
def confidence_level(score):
    if score >= 0.75:
        return "High"
    elif score >= 0.50:
        return "Medium"
    else:
        return "Low"

# ================= UI =================
st.title("Face Recognition System")

mode = st.selectbox(
    "Mode",
    [
        "Image Recognition",
        "Webcam Recognition",
        "Register Face"
    ]
)

# ================= IMAGE RECOGNITION =================
if mode == "Image Recognition":
    uploaded = st.file_uploader("Upload image", ["jpg", "png", "jpeg"])

    if uploaded:
        image = cv2.imdecode(
            np.frombuffer(uploaded.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        results = face_recog.recognize(image)

        table_rows = []

        for idx, ((x1, y1, x2, y2), name, score) in enumerate(results, start=1):

            if score < face_recog.threshold:
                name = "Unknown"
                color = (0, 0, 255)  # RED
            else:
                color = (0, 255, 0)  # GREEN

            label = f"{name} | ID: {idx}"

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

            table_rows.append({
                "Face ID": idx,
                "Name": name,
                "Similarity": round(score, 3),
                "Confidence Level": confidence_level(score)
            })

        st.image(image, channels="BGR")
        recognized_names = sorted({
            row["Name"] for row in table_rows
            if row["Name"] != "Unknown"
        })

        if recognized_names:
            st.subheader(f"Recognized Faces ({len(recognized_names)})")
            for name in recognized_names:
                st.markdown(f"- **{name}**")
        else:
            st.info("No known faces recognized")

        df = pd.DataFrame(table_rows)
        detected = len(results)
        recognized = (df["Name"] != "Unknown").sum()
        unknown = (df["Name"] == "Unknown").sum()
        

        col1, col2, col3 = st.columns(3)
        col1.success(f"Detected faces: {detected}")
        col2.success(f"Recognized faces: {recognized}")
        col3.warning(f"Unknown faces: {unknown}")

        st.subheader("Recognition Results")
        st.dataframe(df, use_container_width=True)

        # st.caption(
        #     "Detected = all faces found by detector. "
        #     "Recognized = similarity ≥ threshold. "
        #     "Unknown = similarity < threshold."
        # 

        # st.caption(
        #     "Similarity is cosine similarity to the face database. "
        #     "Bounding boxes show identity only; details are in the table."
        # )

# ================= WEBCAM RECOGNITION =================
elif mode == "Webcam Recognition":
    start = st.button("▶ Start Webcam")
    stop = st.button("⏹ Stop Webcam")
    frame_box = st.empty()

    if start:
        cap = cv2.VideoCapture(0)
        last_time = time.time()
        fps = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            results = face_recog.recognize(frame)

            for idx, ((x1, y1, x2, y2), name, score) in enumerate(results, start=1):

                if score < face_recog.threshold:
                    name = "Unknown"
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                label = f"{name} | ID: {idx}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1 / max(now - last_time, 1e-6))
            last_time = now

            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

            frame_box.image(frame, channels="BGR")

            if stop:
                break

        cap.release()
        st.success("Webcam stopped")

# ================= REGISTER FACE =================
elif mode == "Register Face":
    name = st.text_input("Name")

    files = st.file_uploader(
        "Upload 3–5 face images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )

    if st.button("Register"):
        if not name or not files or len(files) < 3:
            st.warning("Name + minimum 3 images required")
        else:
            images = [
                cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
                for f in files
            ]

            if face_recog.register(name, images):
                st.success(f"{name} registered successfully")
            else:
                st.error("No face detected in uploaded images")
