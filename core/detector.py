from ultralytics import YOLO
import numpy as np


class FaceDetector:
    def __init__(self, model_path, conf=0.6):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, image):
        results = self.model(image, conf=self.conf, verbose=False)

        boxes = []
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                score = float(box.conf[0])

                boxes.append([
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                    score
                ])

        return boxes
