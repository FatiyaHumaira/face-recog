import streamlit as st
import cv2
import numpy as np
from PIL import Image

from core.detector import FaceDetector
from core.drawer import draw_boxes

MODEL_PATH = "models/yolov8n_100e.pt"

st.set_page_config(page_title="Face Detection", layout="centered")
st.title("YOLOv8 Face Detection")

@st.cache_resource
def load_detector():
    return FaceDetector(MODEL_PATH, conf=0.3)

detector = load_detector()

uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)

    boxes = detector.detect(image)
    st.write(f"Detected faces: {len(boxes)}")

    image_drawn = draw_boxes(image.copy(), boxes)
    st.image(image_drawn, channels="RGB", use_container_width=True)
