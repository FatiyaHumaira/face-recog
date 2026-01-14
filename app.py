import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
from datetime import datetime

from core.face_recog import FaceRecognition

# Path  untuk database wajah
FACE_DB_PATH = "face_db"

@st.cache_resource
def load_model():
    return FaceRecognition(
        db_path=FACE_DB_PATH,
        threshold=0.45   
    )

face_recog = load_model()

st.title("Face Recognition System")

# Fitur-fitur5
mode = st.selectbox(
    "Mode",
    [
        "Image Recognition",
        "Webcam Recognition",
        "Register Face"
    ]
)

# Fitur image recognition
if mode == "Image Recognition":
    uploaded = st.file_uploader(
        "Upload image",
        ["jpg", "png", "jpeg"]
    )

    if uploaded:
        image = cv2.imdecode(
            np.frombuffer(uploaded.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        results = face_recog.recognize(image)

        detected = len(results)
        recognized = 0
        unknown = 0

        table_data = []
        recognized_names = set()

        for idx, ((x1, y1, x2, y2), name, score) in enumerate(results, start=1):
            if name == "Unknown":
                color = (0, 0, 255)   # Bounding box merah
                unknown += 1
            else:
                color = (0, 255, 0)   # Bounding box hijau
                recognized += 1
                recognized_names.add(name)

            label = f"{name} | ID {idx}" # Text pada bounding box

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            # Insert data ke tabel hasil pengenalan
            table_data.append({
                "Face ID": f"ID{idx}",
                "Name": name,
                "Similarity (%)": round(score * 100, 2)
            })


        st.image(image, channels="BGR")
        if recognized_names:
            st.info("Recognized persons: " + ", ".join(recognized_names))
            
        # Info ringkasan pengenalan
        col1, col2, col3 = st.columns(3)

        with col1:
            st.success(f"Detected: {detected}")

        with col2:
            st.success(f"Recognized: {recognized}")

        with col3:
            st.warning(f"Unknown: {unknown}")


        # Menampilkan tabel hasil pengenalan
        if table_data:
            st.subheader("Recognition Result")
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)

# Fitur webcam recognition
elif mode == "Webcam Recognition":
    start = st.button("▶ Start Webcam")
    stop = st.button("⏹ Stop Webcam")

    frame_box = st.empty()
    log_box = st.empty()

    if start:
        cap = cv2.VideoCapture(0)

        recognition_log = {}  # Menyimpan log pengenalan
        last_time = time.time()
        fps = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            results = face_recog.recognize(frame)

            for (x1, y1, x2, y2), name, score in results:
                if name == "Unknown":
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                    # Update recognition log (masih untuk yg dikenali saja)
                    if name not in recognition_log:
                        recognition_log[name] = {
                            "First Seen": datetime.now().strftime("%H:%M:%S"),
                            "Count": 1
                        }
                    else:
                        recognition_log[name]["Count"] += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    name,
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

            # Menampilkan FPS
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

            if recognition_log:
                df_log = (
                    pd.DataFrame.from_dict(recognition_log, orient="index")
                    .reset_index()
                    .rename(columns={"index": "Name"})
                )
                log_box.subheader("Recognition Log")
                log_box.dataframe(df_log, use_container_width=True)

            if stop:
                break

        cap.release()
        st.success("Webcam stopped")

# Fitur register face
elif mode == "Register Face":
    st.subheader("Register Face")

    name = st.text_input("Name")

    method = st.radio(
        "Registration Method",
        ["Upload Images", "Webcam"]
    )

    # Register dengan upload gambar
    if method == "Upload Images":
        files = st.file_uploader(
            "Upload 3–5 foto wajah",
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=True
        )

        if st.button("Register"):
            if not name or not files or len(files) < 3:
                st.warning("Name + minimum 3 images required")
            else:
                images = [
                    cv2.imdecode(
                        np.frombuffer(f.read(), np.uint8),
                        cv2.IMREAD_COLOR
                    )
                    for f in files
                ]

                success, msg = face_recog.register(name, images)

                if success:
                    st.success(msg)
                else:
                    st.error(msg)

    # Register dengan webcam
    else:
        st.info("Start webcam → Capture 3–5 good face frames → Finish")

        if "cam_running" not in st.session_state:
            st.session_state.cam_running = False
        if "frames" not in st.session_state:
            st.session_state.frames = []
        if "cap" not in st.session_state:
            st.session_state.cap = None

        col1, col2 = st.columns([3, 1])

        with col1:
            start = st.button("▶ Start Webcam")
            stop = st.button("⏹ Stop Webcam")
            frame_box = st.empty()

        with col2:
            capture = st.button("📸 Capture")
            finish = st.button("✅ Register")

        # Start webcam
        if start and not st.session_state.cam_running:
            st.session_state.cap = cv2.VideoCapture(0)
            st.session_state.cam_running = True

        # Stop webcam
        if stop and st.session_state.cam_running:
            st.session_state.cam_running = False
            st.session_state.cap.release()

        # Frame loop
        if st.session_state.cam_running:
            ret, frame = st.session_state.cap.read()
            if ret:
                frame = cv2.resize(frame, (640, 480))

                faces = face_recog.model.get(frame)
                for face in faces:
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                frame_box.image(frame, channels="BGR")
                st.session_state.last_frame = frame.copy()

        # Capture frame
        if capture:
            if "last_frame" in st.session_state:
                st.session_state.frames.append(
                    st.session_state.last_frame
                )
                st.success(
                    f"Captured {len(st.session_state.frames)} frame(s)"
                )
            else:
                st.warning("Webcam not ready")

        # Selesaikan pendaftaran
        if finish:
            if not name:
                st.warning("Enter name first")
            elif len(st.session_state.frames) < 3:
                st.warning("Capture at least 3 frames")
            else:
                success, msg = face_recog.register(
                    name,
                    st.session_state.frames
                )

                if success:
                    st.success(f"✅ Registration successful: {name}")

                    st.session_state.frames.clear()
                    st.session_state.cam_running = False

                    if st.session_state.cap:
                        st.session_state.cap.release()

                else:
                    st.error(f"❌ Registration failed: {msg}")


