# core/tracker.py
import cv2

class FaceTracker:
    def __init__(self):
        self.trackers = []
        self.next_id = 0

    def init_from_detections(self, frame, detections):
        self.trackers = []
        for det in detections:
            x1, y1, x2, y2, score = det
            w, h = x2 - x1, y2 - y1
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, (x1, y1, w, h))
            self.trackers.append({
                "id": self.next_id,
                "tracker": tracker,
                "bbox": (x1, y1, x2, y2)
            })
            self.next_id += 1

    def init_or_update(self, frame, detections, iou_thresh=0.3):
        """
        Kalau belum ada tracker → init
        Kalau sudah ada → biarkan update() yang jalan
        """
        if len(self.trackers) == 0 and detections:
            self.init_from_detections(frame, detections)

    def update(self, frame):
        results = []
        for t in self.trackers:
            ok, bbox = t["tracker"].update(frame)
            if ok:
                x, y, w, h = map(int, bbox)
                results.append([x, y, x + w, y + h, t["id"]])
        return results
