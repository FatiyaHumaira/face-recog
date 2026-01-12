import streamlit as st
import cv2
import numpy as np
import time
from core.detector import FaceDetectorYOLO
from core.tracker import FaceTracker
from core.face_recog import FaceRecognition

YOLO_MODEL_PATH = "models/yolov8n_face.onnx"
FACE_DB_PATH = "face_db"

# ================== Load Models ==================
@st.cache_resource
def load_models():
    detector = FaceDetectorYOLO(YOLO_MODEL_PATH, conf_thresh=0.25, iou_thresh=0.3)
    tracker = FaceTracker()
    face_recog = FaceRecognition(db_path=FACE_DB_PATH)
    return detector, tracker, face_recog

detector, tracker, face_recog = load_models()

st.title("Face Recognition System")
mode = st.selectbox("Mode", ["Image", "Webcam", "RTSP CCTV", "Register Face"])

# ==================== Helper ====================
def match_name_to_bbox(frame, detections, face_recog):
    """
    Untuk setiap YOLO bbox, ambil name & confidence dari InsightFace
    yang paling overlap. Tidak pakai tracker ID.
    """
    recog_results = face_recog.recognize(frame)
    matched_results = []

    for det in detections:
        x1, y1, x2, y2, score_det = det
        best_iou = 0
        best_name = "Unknown"
        best_conf = 0

        for bbox, name, conf in recog_results:
            x1_f, y1_f, x2_f, y2_f = bbox
            xi1 = max(x1, x1_f)
            yi1 = max(y1, y1_f)
            xi2 = min(x2, x2_f)
            yi2 = min(y2, y2_f)
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            bbox_area = (x2 - x1) * (y2 - y1)
            iou = inter_area / (bbox_area + 1e-6)
            if iou > best_iou:
                best_iou = iou
                best_name = name
                best_conf = conf

        matched_results.append((det, best_name, best_conf))
    return matched_results

def draw_matched_results(frame, matched_results):
    """
    Menggambar bounding box hijau, nama + confidence + tracker ID
    """
    for i, (det, name, conf) in enumerate(matched_results):
        x1, y1, x2, y2, _ = det
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        text = f"{name} ({conf*100:.2f}%) ID:{i+1}"
        bbox_height = y2 - y1
        fontScale = max(bbox_height / frame.shape[0] * 2.0, 0.5)
        thickness = max(int(bbox_height / frame.shape[0] * 2), 1)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale, thickness)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), (0,0,0), -1)
        cv2.putText(frame, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0,0,255), thickness)
    return frame

def display_face_info(matched_results):
    recognized = []
    unknown_count = 0
    for det, name, conf in matched_results:
        if name == "Unknown":
            unknown_count += 1
        else:
            recognized.append((name, conf))
    st.markdown(f"**Total faces detected:** {len(matched_results)}")
    st.markdown(f"**Known faces:** {len(recognized)}")
    st.markdown(f"**Unknown faces:** {unknown_count}")
    if recognized:
        st.markdown("**Recognized faces:**")
        for name, conf in recognized:
            st.markdown(f"- {name} ({conf*100:.2f}%)")

# ================= IMAGE MODE =================
if mode == "Image":
    uploaded = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
    if uploaded:
        image = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
        detections = detector.detect(image)
        tracker.init_from_detections(image, detections)
        tracks = tracker.update(image)
        matched_results = match_name_to_bbox(image, detections, face_recog)
        image = draw_matched_results(image, matched_results)
        st.image(image, channels="BGR")
        display_face_info(matched_results)

# ================= WEBCAM MODE =================
elif mode == "Webcam":
    start = st.button("Start Webcam")
    stop = st.button("Stop Webcam")
    frame_placeholder = st.empty()
    if start:
        cap = cv2.VideoCapture(0)
        tracker = FaceTracker()
        initialized = False
        frame_count = 0
        last_detections = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 10 == 0 or not initialized:
                last_detections = detector.detect(frame)
                tracker.init_from_detections(frame, last_detections)
                initialized = True

            tracks = tracker.update(frame)
            matched_results = match_name_to_bbox(frame, last_detections, face_recog)
            frame = draw_matched_results(frame, matched_results)
            frame_placeholder.image(frame, channels="BGR")
            display_face_info(matched_results)

            time.sleep(0.03)
            if stop:
                break
        cap.release()

# ================= RTSP CCTV MODE =================
elif mode == "RTSP CCTV":
    rtsp_url = st.text_input("RTSP URL", "rtsp://username:password@ip:554/stream1")
    start = st.button("Start Stream")
    stop = st.button("Stop Stream")
    frame_placeholder = st.empty()
    if start and rtsp_url:
        cap = cv2.VideoCapture(rtsp_url)
        tracker = FaceTracker()
        initialized = False
        frame_count = 0
        last_detections = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 10 == 0 or not initialized:
                last_detections = detector.detect(frame)
                tracker.init_from_detections(frame, last_detections)
                initialized = True

            tracks = tracker.update(frame)
            matched_results = match_name_to_bbox(frame, last_detections, face_recog)
            frame = draw_matched_results(frame, matched_results)
            frame_placeholder.image(frame, channels="BGR")
            display_face_info(matched_results)

            time.sleep(0.03)
            if stop:
                break
        cap.release()

# ================= REGISTER FACE (Upload Multi-Angle) =================
elif mode == "Register Face":
    st.subheader("Register Face (Upload 3-5 images)")

    name_input = st.text_input("Enter name for this face")
    uploaded_files = st.file_uploader("Upload 3-5 face images", type=["jpg","png","jpeg"], accept_multiple_files=True)
    register_btn = st.button("Register Face")

    if register_btn:
        if not name_input.strip():
            st.warning("Please enter a valid name!")
        elif not uploaded_files or len(uploaded_files) < 3:
            st.warning("Please upload at least 3 images for multi-angle registration!")
        else:
            embeddings = []
            for file in uploaded_files:
                img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
                faces = face_recog.model.get(img)
                if faces:
                    embeddings.append(faces[0].embedding)
            if embeddings:
                avg_emb = np.mean(embeddings, axis=0)
                np.save(f"{FACE_DB_PATH}/{name_input.strip()}.npy", avg_emb)
                face_recog.embeddings[name_input.strip()] = avg_emb
                st.success(f"Face '{name_input.strip()}' registered successfully with {len(embeddings)} images!")
            else:
                st.error("No faces detected in the uploaded images!")
