import cv2
import numpy as np

class FaceTracker:
    def __init__(self):
        self.trackers = []  # list of {"id": int, "tracker": Tracker, "bbox": [x,y,w,h]}
        self.next_id = 1

    def iou(self, box1, box2):
        # box = [x1,y1,x2,y2]
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        box1Area = (box1[2]-box1[0])*(box1[3]-box1[1])
        box2Area = (box2[2]-box2[0])*(box2[3]-box2[1])
        iou = interArea / float(box1Area + box2Area - interArea + 1e-5)
        return iou

    def init_or_update(self, frame, detections, iou_thresh=0.3):
        """
        detections: list of [x1,y1,x2,y2,score]
        Hanya buat tracker baru untuk wajah yang belum ada
        """
        updated_trackers = []

        for det in detections:
            x1, y1, x2, y2, score = det
            w, h = x2 - x1, y2 - y1

            # cek apakah sudah ada tracker existing yang overlap
            exists = False
            for t in self.trackers:
                tx1, ty1, tx2, ty2 = t["bbox"]
                if self.iou([x1,y1,x2,y2], [tx1,ty1,tx2,ty2]) > iou_thresh:
                    exists = True
                    break

            if not exists:
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, (x1, y1, w, h))
                self.trackers.append({
                    "id": self.next_id,
                    "tracker": tracker,
                    "bbox": [x1, y1, x2, y2]
                })
                self.next_id += 1

    def update(self, frame):
        results = []
        for t in self.trackers:
            ok, bbox = t["tracker"].update(frame)
            if ok:
                x, y, w, h = map(int, bbox)
                t["bbox"] = [x, y, x+w, y+h]
                results.append([x, y, x+w, y+h, t["id"]])
        return results
