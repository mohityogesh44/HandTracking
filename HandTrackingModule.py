import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode = False, max_hands = 2, detection_confidence = 0.5, track_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.detection_confidence, self.track_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, handlms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_number=0, draw = True):
        landmark_list = []
        if self.results.multi_hand_landmarks:
            #Get the landmarks of the given hand number
            myhand = self.results.multi_hand_landmarks[hand_number]
            #Loop through each landmark of that hand and get coordinates
            for id, landmark in enumerate(myhand.landmark):
                height, width, channel = img.shape
                #Coordinates of center of landmark
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                #Append the coordinates to landmark list
                landmark_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255))
                    cv2.putText(img, str(id), (cx+10, cy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        return landmark_list



def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    prev_time = 0
    curr_time = 0
    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        landmark_list = detector.find_position(img)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__=="__main__":
    main()