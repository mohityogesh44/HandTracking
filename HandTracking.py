import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
prev_time = 0
curr_time = 0
while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    # print(results.multi_hand_landmarks)
    #If there are hands detected in an image
    if results.multi_hand_landmarks:
        #Loop through the landmarks
        for handlms in results.multi_hand_landmarks:
            #Loop through each hands' every landmark and get their coordinates
            for id, landmark in enumerate(handlms.landmark):
                height, width, channel = img.shape

                #Position of center (x and y coordinates)
                #The coordinates are in decimals and we need pixel values, so to get them we multiply x by width and y by height of image.
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                # print(id, cx, cy)

                #Let's try and locate a specific landmark on img
                cv2.circle(img, (cx, cy), 10, (255,0,255))
                cv2.putText(img, str(id), (cx+10, cy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            #Draw hand landmarks on image and connect the landmarks
            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)

    #Calculating FPS
    curr_time = time.time()
    fps = 1/(curr_time - prev_time)
    prev_time = curr_time

    #Showing FPS on screen
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

    #Display Image
    cv2.imshow("Image", img)
    cv2.waitKey(1)