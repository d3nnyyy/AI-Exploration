from ultralytics import YOLO
import cv2 as cv
import cvzone

cap = cv.VideoCapture(0)  # Webcam
cap.set(3, 1280)
cap.set(4, 720)

# cap = cv.VideoCapture("../Videos/bikes.mp4")  # Video

model = YOLO("../Yolo-Weights/yolov8n.pt")

while True:

    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)

            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

    cv.imshow("Image", img)
    cv.waitKey(1)
