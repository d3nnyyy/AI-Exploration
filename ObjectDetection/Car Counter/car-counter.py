import math

from ultralytics import YOLO
import cv2 as cv
import cvzone

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
                cvzone.putTextRect(img, f'{classNames[cls]} {confidence}', (max(0, x1), max(350, y1)), scale=0.7,
                                   thickness=1)
                cvzone.cornerRect(img, (x1, y1, w, h), l=5)

    cv.imshow("Image", img)
    cv.waitKey(0)
