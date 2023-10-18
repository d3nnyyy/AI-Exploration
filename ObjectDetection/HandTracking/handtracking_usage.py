import cv2
import handtracking_module as htm

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Create an instance of the HandTracker class
tracker = htm.HandTracker()

while True:

    # Read a frame from the video
    success, img = cap.read()

    # Find hands in the frame and optionally draw landmarks
    img = tracker.find_hands(img, draw=True)

    # Find and display the landmark positions
    landmarks_list = tracker.find_position(img, draw=False)

    # Print the position of the 12th landmark
    if len(landmarks_list) != 0:
        print(landmarks_list)

    # Display the image
    cv2.imshow("Image", img)
    cv2.waitKey(1)
