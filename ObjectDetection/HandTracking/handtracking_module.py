import cv2 as cv
import mediapipe as mp


# Define a HandTracker class for hand tracking and landmark detection.
class HandTracker:
    """
    A class used to track hands and detect landmarks.
    """

    def __init__(self, mode=False, max_hands=2, model_complexity=1, detection_con=0.5, track_con=0.5):

        """
        Initialize the HandTracker class with specified parameters.

        Args:
            mode (bool): Whether to run in detection mode or tracking mode.
            max_hands (int): Maximum number of hands to detect.
            model_complexity (int): The complexity of the hand model (0, 1, or 2).
            detection_con (float): Minimum confidence threshold for hand detection.
            track_con (float): Minimum confidence threshold for hand tracking.
        """

        # Initialize class attributes with provided parameters.
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_con = detection_con
        self.track_con = track_con

        # Create instances of the MediaPipe hands module and drawing utilities.
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.model_complexity,
                                         self.detection_con, self.track_con)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):

        """
        Find hands in the input image and draw landmarks on the hands if specified.

        Args:
            img (numpy.ndarray): The input image.
            draw (bool): Whether to draw landmarks on the detected hands.

        Returns:
            img (numpy.ndarray): The image with landmarks (if drawn).
        """

        # Convert the input image to RGB format for compatibility with MediaPipe.
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Process the image to detect hands.
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    # Draw landmarks and hand connections on the image.
                    self.mpDraw.draw_landmarks(img, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_no=0, draw=True):

        """
        Find the landmarks' positions on a detected hand.

        Args:
            img (numpy.ndarray): The input image.
            hand_no (int): Index of the hand to track (0 for the first hand).
            draw (bool): Whether to draw landmarks on the detected hand.

        Returns:
            landmarks_list (list): List of landmark positions.
        """

        landmarks_list = []

        if self.results.multi_hand_landmarks:

            # Get the specified hand by index (default is the first hand).
            hand = self.results.multi_hand_landmarks[hand_no]

            for id, landmark in enumerate(hand.landmark):

                h, w, c = img.shape

                # Calculate the landmark's position in image coordinates.
                cx, cy = int(landmark.x * w), int(landmark.y * h)

                landmarks_list.append([id, cx, cy])

                if draw:
                    # Draw circles at the landmark positions on the image.
                    cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)

        return landmarks_list


# Main function for running the hand tracking application.
def main():
    # Create a video capture object to capture video from the default camera (index 0).
    cap = cv.VideoCapture(0)

    # Create an instance of the HandTracker class.
    tracker = HandTracker()

    while True:

        # Read a frame from the video capture.
        success, img = cap.read()

        # Find hands in the frame and optionally draw landmarks.
        img = tracker.find_hands(img)

        # Find and display the landmark positions.
        landmarks_list = tracker.find_position(img)

        if len(landmarks_list) != 0:
            # Print the position of the 12th landmark.
            print(landmarks_list[12])

        cv.waitKey(1)


if __name__ == "__main__":
    main()
