print("Top reached")
import os
import time
# import rtmidi
import pickle
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print ("Right after imports")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode


print("Before main")
def main():
    print("Start main()")
    # Landmarker creation using .task model
    options = HandLandmarkerOptions(
        base_options = BaseOptions(model_asset_path="models/hand_landmarker.task"),
        running_mode = VisionRunningMode.IMAGE, # Per frame mode
        num_hands = 1,
    )
    landmarker = HandLandmarker.create_from_options(options)

    # Try to load the trained data to make predictions
    model_path = "data/gesture_model.pkl"
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run training.py first.")
    
    with open(model_path, "rb") as f:
        gesture_model = pickle.load(f)

    # Open WebCam
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Webcam couldn't be opened.")

    # Midi output instance
    # midiout = rtmidi.MidiOut()
    # ports = midiout.get_ports()
    # Open real midi port else, create a virtual port PD can connect
    
    # print("MIDI output ports:")
    # for i, name in enumerate(ports):
        # print(i, name)
    
    # if ports:
        # midiout.open_port(0)
    # else:
        # midiout.open_virtual_port("HandGestures")

    # Midi variables tracking
    # last_gesture = None
    # last_note = None

    gesture_to_note = {
                    "go_pokes": 60,    # C4
                    "peace": 62,       # D4
                    "thumbs_up": 64,   # E4
                }
                
    VELOCITY = 100 # Not significant
    CHANNEL = 0 # MIDI Channel 1


    while True:
        ret, frame = capture.read()
        if not ret:
            break

        # Convert image to RGB and wrap in mp.image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Reads converted frame into mp_image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Run hand landmark detection
        results = landmarker.detect(mp_image)

        # Draw landmarks if present
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                h, w, _ = frame.shape
                # Draw small green circles on each landmark
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
                # Build feature vector: [x0, y0, z0,...]
                features = []
                for lm in hand_landmarks:
                    features.extend([lm.x, lm.y, lm.z])

                X = np.array(features, dtype=np.float32).reshape(1, -1)

                # Predict label
                gesture_name = gesture_model.predict(X)[0]

                # gesture = str(gesture_name)
                # note = gesture_to_note.get(gesture, None)

                # if note is not None and gesture != last_gesture:
                #     # Change note
                #     if last_note is not None:
                #         midiout.send_message([0x80 | CHANNEL, last_note, 0]) # Note off

                #     # velocity = 100
                #     midiout.send_message([0x90 | CHANNEL, note, VELOCITY]) # Note on

                #     last_gesture = gesture
                #     last_note = note
                

                # Put text near wrist (landmark 0)
                wrist = hand_landmarks[0]
                wx, wy = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(frame, f"Gesture: {gesture_name}",
                            (wx - 40, wy -20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            
            # if last_note is not None:
            #     midiout.send_message([0x80 | CHANNEL, last_note, 0]) # Note off
            #     last_note = None
            #     last_gesture = None

            cv2.putText(frame, "No hand detected",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        # Show frame
        cv2.imshow("Hand Tracking (Tasks API)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean up and closing

    # # Close midi resources
    # if last_note is not None:
    #     midiout.send_message([0x80, last_note, 0]) # Note off
    # del midiout


    # Close computer vision sources
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    