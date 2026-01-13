import streamlit as st
import cv2
import numpy as np
import os
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from core.detector import FaceDetectorYOLO
from core.tracker import FaceTracker
from core.drawer import draw_boxes, draw_tracks
from core.face_recog import FaceRecognition
from core.webcam_register import register_from_webcam

# ===================== CONFIG =====================
YOLO_MODEL_PATH = "models/yolov8n_face.onnx"
FACE_DB_PATH = "face_db"

# ===================== LOAD MODELS =====================
@st.cache_resource
def load_models():
    detector = FaceDetectorYOLO(YOLO_MODEL_PATH, conf_thresh=0.25, iou_thresh=0.3)
    tracker = FaceTracker()
    face_recog = FaceRecognition(db_path=FACE_DB_PATH)
    return detector, tracker, face_recog

detector, tracker, face_recog = load_models()

# ===================== UI =====================
st.title("Face Recognition System")

mode = st.selectbox(
    "Mode",
    [
        "Image Recognition",
        "Webcam Recognition",
        "Register Face",
        "RTSP CCTV"
    ]
)

# ===================== IMAGE RECOGNITION =====================
if mode == "Image Recognition":
    uploaded = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
    if uploaded:
        image = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)

        detections = detector.detect(image)
        image = draw_boxes(image, detections)

        results = face_recog.recognize(image)

        recognized = 0
        unknown = 0
        conf_scores = []
        table_data = []

        for bbox, name, score in results:
            x1, y1, x2, y2 = bbox
            label = f"{name} ({score*100:.1f}%)"

            cv2.putText(
                image,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            if name == "Unknown":
                unknown += 1
            else:
                recognized += 1
                conf_scores.append(score)

            table_data.append({
                "Name": name,
                "Confidence (%)": round(score * 100, 2)
            })

        st.image(image, channels="BGR")

        st.success(f"Detected faces (YOLO): {len(detections)}")
        st.success(f"Recognized faces: {recognized}")
        st.warning(f"Unknown faces: {unknown}")

# ===================== WEBCAM RECOGNITION =====================
elif mode == "Webcam Recognition":

    start = st.button("▶ Start Webcam")
    stop = st.button("⏹ Stop Webcam")
    frame_placeholder = st.empty()

    if start:
        cap = cv2.VideoCapture(0)
        tracker = FaceTracker()

        frame_count = 0
        detections = []
        track_id_to_name = {}

        DETECT_EVERY = 12
        RECOG_EVERY = 20

        last_time = time.time()
        fps = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            frame_count += 1

            # ---------- DETECTION ----------
            if frame_count % DETECT_EVERY == 0:
                detections = detector.detect(frame)
                tracker.init_or_update(frame, detections)

            tracks = tracker.update(frame)

            # ---------- RECOGNITION (CACHE) ----------
            if frame_count % RECOG_EVERY == 0:
                recog_results = face_recog.recognize(frame)

                for x1,y1,x2,y2,tid in tracks:
                    for fb,name,score in recog_results:
                        fx1,fy1,fx2,fy2 = fb
                        ix1 = max(x1, fx1)
                        iy1 = max(y1, fy1)
                        ix2 = min(x2, fx2)
                        iy2 = min(y2, fy2)
                        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                        area = (fx2-fx1)*(fy2-fy1)

                        if area > 0 and inter/area > 0.5:
                            track_id_to_name[tid] = (name, score)
                            break

            # ---------- DRAW ----------
            for x1,y1,x2,y2,tid in tracks:
                name, score = track_id_to_name.get(tid, ("Unknown", 0))
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(
                    frame,
                    f"{name} ({score*100:.1f}%)",
                    (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

            # ---------- FPS ----------
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1 / max(now-last_time, 1e-6))
            last_time = now
            cv2.putText(frame, f"FPS: {int(fps)}", (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

            frame_placeholder.image(frame, channels="BGR")

            if stop:
                break

        cap.release()
        st.success("Webcam stopped")

# ===================== REGISTER FACE =====================
elif mode == "Register Face":
    st.subheader("Register Face")

    name_input = st.text_input("Name")

    method = st.radio(
        "Method",
        ["Upload Images", "Webcam (Inline)"]
    )

    # ================= UPLOAD =================
    if method == "Upload Images":
        files = st.file_uploader(
            "Upload 3–5 images",
            type=["jpg","png","jpeg"],
            accept_multiple_files=True
        )

        if st.button("Register"):
            if not name_input or not files or len(files) < 3:
                st.warning("Name + minimum 3 images required")
            else:
                embs = []
                for f in files:
                    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
                    faces = face_recog.model.get(img)
                    if faces:
                        embs.append(faces[0].embedding)

                if embs:
                    avg = np.mean(embs, axis=0)
                    np.save(f"{FACE_DB_PATH}/{name_input}.npy", avg)
                    face_recog.embeddings[name_input] = avg
                    st.success(f"{name_input} registered")
                else:
                    st.error("No face detected")

    # ================= WEBCAM INLINE =================
    else:
        st.info("Klik Start Webcam → Capture 3–5 kali → Finish")

        # session state init
        if "cam_running" not in st.session_state:
            st.session_state.cam_running = False
        if "last_frame" not in st.session_state:
            st.session_state.last_frame = None
        if "captured_frames" not in st.session_state:
            st.session_state.captured_frames = []

        col1, col2 = st.columns([3, 1])

        with col1:
            start = st.button("▶ Start Webcam")
            stop = st.button("⏹ Stop Webcam")
            frame_box = st.empty()

        with col2:
            capture = st.button("📸 Capture")
            finish = st.button("✅ Finish")

        # START
        if start:
            st.session_state.cam_running = True
            st.session_state.cap = cv2.VideoCapture(0)

        # STOP
        if stop and "cap" in st.session_state:
            st.session_state.cam_running = False
            st.session_state.cap.release()

        # LOOP FRAME
        if st.session_state.cam_running:
            ret, frame = st.session_state.cap.read()
            if ret:
                frame = cv2.resize(frame, (640, 480))
                st.session_state.last_frame = frame.copy()

                faces = face_recog.model.get(frame)
                for face in faces:
                    x1,y1,x2,y2 = face.bbox.astype(int)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                frame_box.image(frame, channels="BGR")

        # CAPTURE
        if capture:
            if st.session_state.last_frame is not None:
                st.session_state.captured_frames.append(
                    st.session_state.last_frame.copy()
                )
                st.success(f"Captured {len(st.session_state.captured_frames)} frame(s)")
            else:
                st.warning("Webcam not ready")

        # FINISH
        if finish:
            if not name_input:
                st.warning("Enter name first")
            elif len(st.session_state.captured_frames) < 3:
                st.warning("Capture at least 3 frames")
            else:
                embs = []
                for img in st.session_state.captured_frames:
                    faces = face_recog.model.get(img)
                    if faces:
                        embs.append(faces[0].embedding)

                if embs:
                    avg = np.mean(embs, axis=0)
                    np.save(f"{FACE_DB_PATH}/{name_input}.npy", avg)
                    face_recog.embeddings[name_input] = avg
                    st.success(f"{name_input} registered successfully")

                    st.session_state.captured_frames.clear()
                    st.session_state.cam_running = False
                    if "cap" in st.session_state:
                        st.session_state.cap.release()
                else:
                    st.error("No face detected")

# # ===================== RTSP CCTV =====================
# elif mode == "RTSP CCTV":
#     rtsp_url = st.text_input("RTSP URL")
#     start = st.button("Start")

#     if start:
#         cap = cv2.VideoCapture(rtsp_url)
#         tracker = FaceTracker()
#         frame_placeholder = st.empty()

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             detections = detector.detect(frame)
#             if detections:
#                 tracker.init_from_detections(frame, detections)

#             tracks = tracker.update(frame)
#             frame = draw_tracks(frame, tracks)

#             results = face_recog.recognize(frame)
#             for bbox, name, score in results:
#                 x1, y1, x2, y2 = bbox
#                 cv2.putText(frame, f"{name} ({score*100:.1f}%)",
#                             (x1, y1 - 8),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

#             frame_placeholder.image(frame, channels="BGR")
#             time.sleep(0.03)

#         cap.release()
