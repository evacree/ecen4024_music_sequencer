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
MIDI_CH   = 0      #channel
VEL       = 100   #volume

n = 60
default_n = 60
notecount = 0

GESTURE_TO_NOTE = {
    "ok": 60,          #  major scale
    "thumbs_up": 62,   # 
    "one" : 64,
    "two" : 65,
    "three" : 67,
    "four" : 69,
    "five" : 71,
    "six" : 72, #end
    "seven" : 76, #extra notes
    "eight" : 78,
    "nine" : 80,
    "ten" : 82
}



outport = mido.open_output(PORT_NAME)
#print(mido.get_output_names())

def midi_panic(outport):#in case of crash
    
    for ch in range(16):
        outport.send(mido.Message('control_change', channel=ch, control=120, value=0))  # All Sound Off
        outport.send(mido.Message('control_change', channel=ch, control=123, value=0))  # All Notes Off
        outport.send(mido.Message('control_change', channel=ch, control=121, value=0))  # Reset All Controllers

def note_on(note, vel):
    vel = int(np.clip(vel, 1, 127))
    outport.send(mido.Message("note_on", note=note, velocity=vel, channel=MIDI_CH))

def note_off(note):
    outport.send(mido.Message("note_off", note=note, velocity=0, channel=MIDI_CH))

def safe_note_off(outport, note, ch):
    outport.send(mido.Message('note_off', note=note, velocity=0, channel=ch))

def set_volume(vol):
    vol = int(np.clip(vol, 0, 127))
    outport.send(mido.Message("control_change", channel=MIDI_CH, control=11, value=vol))

def map_range(val, in_min, in_max, out_min, out_max):
    val = np.clip(val, in_min, in_max)
    return out_min + (val - in_min) * (out_max - out_min) / (in_max - in_min)

MIN_NOTE = 60
MAX_NOTE = 72


MODEL_PATH = "gesture_model.pkl"

with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Windows DirectShow chosen
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)



mp_drawing = mp.solutions.drawing_utils 
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4)

STABLE_ON  = 4   #frames of a gesture before note turns on
STABLE_OFF = 4   #frames w/o gesture for note to turn off

candidate = None
cand_count = 0

stable_gesture = None
none_count = 0
current_note = None
last_volume = -1
current_velocity = None

#latency tracking
pending_on_label = None
pending_on_frames = 0

pending_off_frames = 0
last_stable_gesture = None

stable_on_duration = 0  # how long the confirmed gesture stayed ON (frames)

smooth_x = 0.5
smooth_y = 0.5
alpha = 0.2

VEL_THRESH = 4   # how much volume must change before retrigger
NOTE_THRESH = 1  # how much note must change before retrigger

#loop
try:

    while True:
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if success:
            RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # RGB conversion
            result = hand.process(RGB_frame)
            hand_label = None
            gesture = None
            right_ten_now = False
            target_note = None
            target_vol = 0
            theremin = False
            

            if result.multi_hand_landmarks and result.multi_handedness:
                h, w = frame.shape[:2]
                for hand_landmarks, handed in zip(result.multi_hand_landmarks, result.multi_handedness):
                    hand_label = handed.classification[0].label

                    coords = []
                    for lm in hand_landmarks.landmark:
                        coords.extend([lm.x, lm.y, lm.z])
                    gesture = data.predict([coords])[0]

                    wrist = hand_landmarks.landmark[0]
                    right_index_tip = hand_landmarks.landmark[8]
                    xpos = right_index_tip.x
                    ypos = right_index_tip.y
                    cv2.circle(frame, (int(xpos * w), int(ypos * h)), 10, (0, 0, 255), -1)
                    smooth_x = (1 - alpha) * smooth_x + alpha * xpos
                    smooth_y = (1 - alpha) * smooth_y + alpha * ypos

                    

                    if hand_label == "Right":
                        theremin = True

                         # x controls note
                        note_float = map_range(smooth_x, 0.0, 1.0, MIN_NOTE, MAX_NOTE)
                        target_note = int(round(note_float))


                        # y controls volume (inverted: top=loud, bottom=quiet)
                        vel_float = map_range(smooth_y, 0.0, 1.0, 127, 1)
                        target_velocity = int(round(vel_float))
                        print("smooth_x:", round(smooth_x, 3),
                        "smooth_y:", round(smooth_y, 3),
                        "note:", target_note,
                        "vol:", target_velocity)


                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if theremin and target_note is not None:
                need_new_note = False
                if current_note is None:
                    need_new_note = True
                elif abs(target_note - current_note) >= NOTE_THRESH:
                    need_new_note = True
                elif current_velocity is None or abs(target_velocity - current_velocity) >= VEL_THRESH:
                    need_new_note = True

                if need_new_note:
                    if current_note is not None:
                        note_off(current_note)
                    note_on(target_note, target_velocity)
                    current_note = target_note
                    current_velocity = target_velocity

            else:
                if current_note is not None:
                    note_off(current_note)
                    current_note = None
                    current_velocity = None
                                
            

            
            cv2.imshow("capture image", frame)
            if cv2.waitKey(1) == ord('q'): # press q to quit
                break
            time.sleep(0.005)

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



