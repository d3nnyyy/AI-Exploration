import math

import numpy
from ultralytics import YOLO
import cv2 as cv
import cvzone
from sort import *

cap = cv.VideoCapture("../Videos/cars.mp4")  # Video

model = YOLO("../Yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv.imread("mask.png")

# Tracking

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

while True:

    success, img = cap.read()

    imgRegion = cv.bitwise_and(img, mask)

    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)

            w, h = x2 - x1, y2 - y1

            # Confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if confidence > 0.3 \
                    and currentClass == "car" \
                    or currentClass == "truck" \
                    or currentClass == "motorbike" \
                    or currentClass == "bus" \
                    or currentClass == "bicycle":
                # cvzone.putTextRect(img, f'{classNames[cls]} {confidence}', (max(0, x1), max(350, y1)), scale=0.7,
                #                    thickness=1)
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, confidence])
                detections = np.vstack((detections, currentArray))

    results_tracker = tracker.update(detections)

    for result in results_tracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=2,
                           thickness=3, offset=10)

    cv.imshow("Image", img)
    cv.waitKey(0)
