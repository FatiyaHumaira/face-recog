from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

YOLO_MODEL_PATH = BASE_DIR / "models" / "yolov8n_face.onnx"

IMG_SIZE = 640
CONF_THRESHOLD = 0.3
NMS_THRESHOLD = 0.3


