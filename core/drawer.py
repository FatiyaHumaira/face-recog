import cv2

def draw_boxes(image, boxes):
    """
    Menggambar bounding box deteksi wajah dengan score/confidence
    """
    img_h, img_w = image.shape[:2]
    
    for x1, y1, x2, y2, score in boxes:
        # Gambar bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Hitung tinggi bbox untuk font proporsional
        bbox_height = y2 - y1
        fontScale = max(bbox_height / img_h * 2.0, 0.5)
        thickness = max(int(bbox_height / img_h * 2), 1)

        # Score teks
        text = f"{score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale, thickness)
        cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1), (0, 0, 0), -1)
        cv2.putText(image, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 255, 0), thickness)
    
    return image


def draw_tracks(image, tracks):
    """
    Menggambar tracker ID di bawah bounding box
    """
    img_h, img_w = image.shape[:2]
    
    for x1, y1, x2, y2, track_id in tracks:
        # Gambar bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Tinggi bbox untuk font proporsional
        bbox_height = y2 - y1
        fontScale = max(bbox_height / img_h * 1.5, 0.5)
        thickness = max(int(bbox_height / img_h * 2), 1)

        # Teks ID tracker
        text = f"ID {track_id}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale, thickness)
        cv2.rectangle(image, (x1, y2), (x1 + tw, y2 + th + 4), (0, 0, 0), -1)
        cv2.putText(image, text, (x1, y2 + th), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 0, 0), thickness)
    
    return image
