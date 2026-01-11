import streamlit as st
import cv2
import numpy as np
import time

from core.detector import FaceDetectorYOLO
from core.tracker import FaceTracker

YOLO_MODEL_PATH = "models/yolov8n_face.onnx"

@st.cache_resource
def load_models():
    detector = FaceDetectorYOLO(YOLO_MODEL_PATH, conf_thresh=0.25, iou_thresh=0.3)
    tracker = FaceTracker()
    return detector, tracker

detector, tracker = load_models()

st.title("Face Detection System")

mode = st.selectbox("Mode", ["Image", "Webcam"])

# ================= IMAGE MODE =================
if mode == "Image":
    uploaded = st.file_uploader("Upload image", type=["jpg","png","jpeg"])
    if uploaded:
        image = cv2.imdecode(
            np.frombuffer(uploaded.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        # Reset tracker untuk image
        tracker = FaceTracker()
        detections = detector.detect(image)
        tracker.init_or_update(image, detections)

        tracks = tracker.update(image)
        for x1, y1, x2, y2, tid in tracks:
            # Ambil confidence dari detection yang overlap tracker
            conf = 0
            for det in detections:
                dx1, dy1, dx2, dy2, score = det
                # Cek overlap > 0.5
                ix1 = max(x1, dx1)
                iy1 = max(y1, dy1)
                ix2 = min(x2, dx2)
                iy2 = min(y2, dy2)
                inter_area = max(0, ix2-ix1) * max(0, iy2-iy1)
                det_area = (dx2-dx1)*(dy2-dy1)
                if det_area > 0 and inter_area/det_area > 0.5:
                    conf = score
                    break

            cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(image, f"{conf:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        st.image(image, channels="BGR")
        st.success(f"Detected faces: {len(tracks)}")


# ================= WEBCAM MODE =================
else:
    start = st.button("Start Webcam")
    stop = st.button("Stop Webcam")
    frame_placeholder = st.empty()
    frame_count = 0

    if start:
        cap = cv2.VideoCapture(0)
        tracker = FaceTracker()

        while cap.isOpened():
            ret, frame = cap.read()
            frame_count += 1
            if not ret:
                break

            # Deteksi ulang setiap 10 frame untuk wajah baru
            if frame_count % 10 == 0:
                detections = detector.detect(frame)
                tracker.init_or_update(frame, detections, iou_thresh=0.3)

            tracks = tracker.update(frame)
            for t in tracks:
                x1, y1, x2, y2, tid = t
                # Ambil confidence dari detection yang overlap tracker
                conf = 0
                if 'detections' in locals():
                    for det in detections:
                        dx1, dy1, dx2, dy2, score = det
                        ix1 = max(x1, dx1)
                        iy1 = max(y1, dy1)
                        ix2 = min(x2, dx2)
                        iy2 = min(y2, dy2)
                        inter_area = max(0, ix2-ix1) * max(0, iy2-iy1)
                        det_area = (dx2-dx1)*(dy2-dy1)
                        if det_area > 0 and inter_area/det_area > 0.5:
                            conf = score
                            break

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{conf:.2f}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            frame_placeholder.image(frame, channels="BGR")
            time.sleep(0.03)

            if stop:
                break

        cap.release()
        st.success("Webcam stopped")
