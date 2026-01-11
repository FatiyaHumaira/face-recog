import cv2
import numpy as np
import onnxruntime as ort

class FaceDetectorYOLO:
    def __init__(self, model_path, conf_thresh=0.25, iou_thresh=0.3):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

    def preprocess(self, image):
        img = cv2.resize(image, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))
        img = np.expand_dims(img, axis=0)
        return img

    def detect(self, image):
        h, w = image.shape[:2]
        input_tensor = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_tensor})[0]
        outputs = outputs[0].T  # (num_boxes, 5)

        boxes, scores = [], []
        for det in outputs:
            score = det[4]
            if score < self.conf_thresh:
                continue
            cx, cy, bw, bh = det[:4]
            x1 = int((cx - bw/2) * w / 640)
            y1 = int((cy - bh/2) * h / 640)
            x2 = int((cx + bw/2) * w / 640)
            y2 = int((cy + bh/2) * h / 640)
            boxes.append([x1, y1, x2-x1, y2-y1])  # NMS expects x,y,w,h
            scores.append(float(score))

        if len(boxes) == 0:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thresh, self.iou_thresh)

        results = []
        for i in indices.flatten():
            x, y, w_box, h_box = boxes[i]
            results.append([x, y, x+w_box, y+h_box, scores[i]])

        return results
