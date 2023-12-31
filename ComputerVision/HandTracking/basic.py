import cv2 as cv
import mediapipe as mp

# Using webcam to capture video
cap = cv.VideoCapture(0)

# Initialize the mediapipe hands module
mpHands = mp.solutions.hands

# Initialize the hands object
hands = mpHands.Hands()

# Initialize the mediapipe drawing utilities module
mpDraw = mp.solutions.drawing_utils

while True:

    # Read a frame from the video
    success, img = cap.read()

    # Convert the image to RGB
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Process the image using mediapipe hands
    results = hands.process(imgRGB)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Print the hand landmarks
            for id, lm in enumerate(hand_landmarks.landmark):
                # Get the height, width, and number of channels of the image
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print("ID: ", id, "x: ", cx, "y: ", cy)

                cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)

            # Draw the hand landmarks
            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

    cv.imshow("Image", img)
    cv.waitKey(1)
