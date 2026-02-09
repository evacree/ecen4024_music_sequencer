import cv2
import numpy as np
import time
import mido
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"SymbolDatabase\.GetPrototype\(\) is deprecated\.",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names",
    category=UserWarning,
)
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import pickle
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

PORT_NAME = "MediaPipe_to_PureData 1"
MIDI_CH   = 0                         # 0 = channel 1
NOTE      = 60                        # middle C
VEL       = 100   
outport = mido.open_output(PORT_NAME)
print(mido.get_output_names())

MODEL_PATH = "gesture_model.pkl"

with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

mp_drawing = mp.solutions.drawing_utils # Imports mediapipe solutions and vision
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4)


while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if success:
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # RGB conversion
        result = hand.process(RGB_frame)
        right_ok_now = False

        if result.multi_hand_landmarks and result.multi_handedness:
            h, w = frame.shape[:2]
            for hand_landmarks, handed in zip(result.multi_hand_landmarks, result.multi_handedness):
                hand_label = handed.classification[0].label

                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                gesture = data.predict([coords])[0]

                wrist = hand_landmarks.landmark[0]
                px, py = int(wrist.x * w), int(wrist.y * h)

                cv2.putText(frame, f"{hand_label}: {gesture}", (px, py -10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print(f"{hand_label} = {gesture}")
                if hand_label == 'Right' and gesture == 'ok':
                    right_ok_now = True

                else:
                    print()

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if right_ok_now and not right_ok_active:
            outport.send(mido.Message('note_on', note=NOTE, velocity=VEL, channel=MIDI_CH))
            print("NOTE ON")
            outport.send(note_on)
            time.sleep(0.15)
            outport.send(note_off)

        

        right_ok_active = right_ok_now
        cv2.imshow("capture image", frame)
        if cv2.waitKey(1) == ord('q'): # If there is a key in this case q quits the program
            break

outport.send(mido.Message('note_off', note=NOTE, velocity=0, channel=MIDI_CH))
outport.close()
cap.release()
cv2.destroyAllWindows()    

#gesture list:
#
#ok
#thumbs_up
#1-10

