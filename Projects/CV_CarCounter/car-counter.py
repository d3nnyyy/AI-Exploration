import math

from ultralytics import YOLO
import cv2 as cv
import cvzone
from sort import *

# Initialize the video capture with the path to the video file.
cap = cv.VideoCapture("Videos/cars.mp4")

# Initialize the YOLO model with the pre-trained weights.
model = YOLO("Yolo-Weights/yolov8n.pt")

# Define a list of class names for object detection.
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

# Load a mask image for region of interest.
mask = cv.imread("mask.png")

# Initialize a SORT tracker.
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Define region of interest limits and an empty list to track detected objects.
limits = [340, 350, 673, 350]
total_count = []

while True:

    # Read a frame from the video.
    success, img = cap.read()
    imgRegion = cv.bitwise_and(img, mask)

    # Overlay the car counter image on the frame.
    imgGraphics = cv.imread("graphics.png", cv.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))

    # Perform object detection using YOLO.
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Extract bounding box coordinates.
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Calculate confidence and class name.
            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Check if the detected object is a car, truck, motorbike, bus, or bicycle with confidence above 0.3.
            if confidence > 0.3 and currentClass in ["car", "truck", "motorbike", "bus", "bicycle"]:
                currentArray = np.array([x1, y1, x2, y2, confidence])
                detections = np.vstack((detections, currentArray))

    # Update object tracking with SORT.
    results_tracker = tracker.update(detections)

    # Draw a line and track objects within specified limits.
    cv.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in results_tracker:

        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw a rectangle and label the tracked object.
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)

        # Calculate the center of the object.
        cx, cy = x1 + w // 2, y1 + h // 2

        # Count and mark objects that cross the specified limits.
        if limits[0] < cx < limits[2] and limits[1] - 30 < cy < limits[1] + 30:
            if total_count.count(id) == 0:
                total_count.append(id)
                cv.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # Display the count of cars found.
    cv.putText(img, str(len(total_count)), (255, 100), cv.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    cv.imshow("Image", img)
    cv.waitKey(1)
