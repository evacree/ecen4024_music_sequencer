import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import pickle
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

MODEL_PATH = "gesture_model.pkl"

with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if success:
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hand.process(RGB_frame)
        if result.multi_hand_landmarks :
            for hand_landmarks in result.multi_hand_landmarks:

                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                gesture = data.predict([coords])[0]
                cv2.putText(frame, gesture, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow("capture image", frame)
        if cv2.waitKey(1) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()    

