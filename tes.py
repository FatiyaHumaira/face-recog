import cv2
print(cv2.__version__)
tracker = cv2.TrackerCSRT_create()
print("CSRT tracker available")
