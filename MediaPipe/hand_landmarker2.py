import cv2
import numpy as np
import time
import mido
import os
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
MIDI_CH   = 0      #channel
VEL       = 100   #volume

GESTURE_TO_NOTE = {
    "ok": 60,          #  major scale   # 
    "one" : 64,
    "two" : 65,
    "three" : 67,
    "four" : 69,
    "five" : 71,
    "six" : 72, #end
    "fist" : 62
}



outport = mido.open_output(PORT_NAME)
#print(mido.get_output_names())

def midi_panic(outport):#in case of crash
    
    for ch in range(16):
        outport.send(mido.Message('control_change', channel=ch, control=120, value=0))  # All Sound Off
        outport.send(mido.Message('control_change', channel=ch, control=123, value=0))  # All Notes Off
        outport.send(mido.Message('control_change', channel=ch, control=121, value=0))  # Reset All Controllers

def note_on(note):
    outport.send(mido.Message("note_on", note=note, velocity=VEL, channel=MIDI_CH))

def note_off(note):
    outport.send(mido.Message("note_off", note=note, velocity=0, channel=MIDI_CH))

def safe_note_off(outport, note, ch):
    outport.send(mido.Message('note_off', note=note, velocity=0, channel=ch))


MODEL_PATH = os.path.join(os.path.dirname(__file__), "gesture_model.pkl")

with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Windows DirectShow chosen
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)



mp_drawing = mp.solutions.drawing_utils 
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

STABLE_ON  = 4   #frames of a gesture before note turns on
STABLE_OFF = 4   #frames w/o gesture for note to turn off

candidate = None
cand_count = 0

stable_gesture = None
none_count = 0
current_note = None

#latency tracking
pending_on_label = None
pending_on_frames = 0

pending_off_frames = 0
last_stable_gesture = None

stable_on_duration = 0  # how long the confirmed gesture stayed ON (frames)


#loop
try:

    while True:
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if success:
            RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # RGB conversion
            result = hand.process(RGB_frame)
            right_ok_now = False
            
            right_one_now = False
            right_two_now = False
            right_three_now = False
            right_four_now = False
            right_five_now = False
            right_six_now = False
            
            right_fist_now = False

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

                    text = f"{gesture}: {target_note}"
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    x = frame.shape[1] - text_width - 10
                    y = frame.shape[0] - 10
                    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    if hand_label == 'Right' and gesture == 'ok':
                        right_ok_now = True
                    
                    if hand_label == 'Right' and gesture == 'one':
                        right_one_now = True
                    if hand_label == 'Right' and gesture == 'two':
                        right_two_now = True
                    if hand_label == 'Right' and gesture == 'three':
                        right_three_now = True
                    if hand_label == 'Right' and gesture == 'four':
                        right_four_now = True
                    if hand_label == 'Right' and gesture == 'five':
                        right_five_now = True
                    if hand_label == 'Right' and gesture == 'six':
                        right_six_now = True
                    
                    if hand_label == 'Right' and gesture == 'fist':
                        right_fist_now = True

                    

            active_gesture = None
            
            if right_ok_now:
                active_gesture = "ok"
            if right_one_now:
                active_gesture = "one"
            if right_two_now:
                active_gesture = "two"
            if right_three_now:
                active_gesture = "three"
            if right_four_now:
                active_gesture = "four"
            if right_five_now:
                active_gesture = "five"
            if right_six_now:
                active_gesture = "six"
            
            if right_fist_now:
                active_gesture = "fist"

            target_note = GESTURE_TO_NOTE.get(active_gesture)  

            if target_note != current_note:
                if current_note is not None:
                    note_off(current_note)
                if target_note is not None:
                    note_on(target_note)
                current_note = target_note
            
            if active_gesture == candidate:
                cand_count += 1
            else:
                candidate = active_gesture
                cand_count = 1

            # debounce
            if candidate is None:
                none_count += 1
                if none_count >= STABLE_OFF:
                    stable_gesture = None
            else:
                none_count = 0
                if cand_count >= STABLE_ON:
                    stable_gesture = candidate

            #latency tracking again

            #ON
            if stable_gesture is None:
                
                pending_on_label = candidate
                pending_on_frames = cand_count if candidate is not None else 0
            else:
        
                stable_on_duration += 1

            
            if stable_gesture != last_stable_gesture:

                # OFF
                if last_stable_gesture is not None and stable_gesture is None:
                    # you only become None after none_count >= STABLE_OFF
                    stable_on_duration = 0

                # ON transition
                if last_stable_gesture is None and stable_gesture is not None:
                    last_stable_gesture = stable_gesture



            target_note = GESTURE_TO_NOTE.get(stable_gesture)

            if target_note != current_note:
                if current_note is not None:
                    note_off(current_note)
                if target_note is not None:
                    note_on(target_note)
                current_note = target_note
                            

            
            cv2.imshow("capture image", frame)
            if cv2.waitKey(1) == ord('q'): # press q to quit
                break

finally:
    # executed on close
    try:
        if current_note is not None:
            note_off(current_note)
        midi_panic(outport)
        outport.close()
    except Exception as e:
        print("MIDI cleanup error")
    cap.release()
    cv2.destroyAllWindows()


outport.send(mido.Message('note_off', note=current_note, velocity=0, channel=MIDI_CH))
outport.close()
cap.release()
cv2.destroyAllWindows()    

