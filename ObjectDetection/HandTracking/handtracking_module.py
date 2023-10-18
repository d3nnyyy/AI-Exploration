import cv2 as cv
import mediapipe as mp


class HandTracker:

    def __init__(self, mode=False, maxHands=2, model_complexity=1, detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity, self.detectionCon,
                                        self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):

        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_no=0, draw=True):

        landmarks_list = []

        if self.results.multi_hand_landmarks:

            hand = self.results.multi_hand_landmarks[hand_no]

            for id, landmark in enumerate(hand.landmark):

                h, w, c = img.shape

                cx, cy = int(landmark.x * w), int(landmark.y * h)

                landmarks_list.append([id, cx, cy])

                if draw:
                    cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)

        return landmarks_list


def main():

    cap = cv.VideoCapture(0)
    tracker = HandTracker()

    while True:

        success, img = cap.read()

        img = tracker.find_hands(img)
        landmarks_list = tracker.find_position(img)

        if len(landmarks_list) != 0:
            print(landmarks_list[4])

        cv.waitKey(1)


if __name__ == "__main__":
    main()
