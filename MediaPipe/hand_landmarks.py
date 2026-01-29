import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          
    (0, 5), (5, 6), (6, 7), (7, 8),          
    (5, 9), (9,10), (10,11), (11,12),        
    (9,13), (13,14), (14,15), (15,16),       
    (13,17), (17,18), (18,19), (19,20),      
    (0,17)                                   
]

MODEL_PATH = "hand_landmarker.task"

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands

with HandLandmarker.create_from_options(options) as landmarker:
    timestamp_ms = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        timestamp_ms += 33
        COLORS = [(255, 0, 0), (0, 0, 255)]
        if result.hand_landmarks:
            h, w, _ = frame.shape
            for i, hand in enumerate(result.hand_landmarks):
                color = COLORS[i % len(COLORS)]

                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

                
                for (x, y) in pts:                                          #points
                    cv2.circle(frame, (x, y), 4, color, -1)

                
                for a, b in HAND_CONNECTIONS:                               #lines
                    cv2.line(frame, pts[a], pts[b], color, 2)

           
        cv2.imshow("hands", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    


cap.release()
cv2.destroyAllWindows()
