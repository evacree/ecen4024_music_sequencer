import os
import time
import pickle
import warnings

import cv2
import mido
import mediapipe as mp

# import sklearn  # optional, not required in code but required as a dependency

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

PORT_NAME = "MediaPipe_to_PureData 1"
MIDI_CH = 0          # 0 = MIDI channel 1
VEL = 100            # velocity

GESTURE_TO_NOTE = {
    "ok": 60,         # C4 (Middle C is 60)
    "thumbs_up": 62,  # D4
    "one": 64,        # E4
    "two": 65,        # F4
    "three": 67,      # G4
    "four": 69,       # A4
    "five": 71,       # B4
    "six": 72,        # C5
    "seven": 76,      # E5
    "eight": 78,      # F#5
    "nine": 80,       # G#5
    "ten": 82,        # A#5
}

# Debounce/stability
STABLE_ON = 4   # frames required before accepting a new gesture
STABLE_OFF = 4  # frames required before releasing to None

# Camera settings (tune down for speed)
FRAME_W = 600
FRAME_H = 500


def midi_panic(outport_obj: mido.ports.BaseOutput) -> None:
    """Stop any stuck notes on all channels."""
    for ch in range(16):
        outport_obj.send(mido.Message('control_change', channel=ch, control=120, value=0))  # All Sound Off
        outport_obj.send(mido.Message('control_change', channel=ch, control=123, value=0))  # All Notes Off
        outport_obj.send(mido.Message('control_change', channel=ch, control=121, value=0))  # Reset All Controllers


def send_note_on(outport_obj: mido.ports.BaseOutput, note: int) -> None:
    outport_obj.send(mido.Message("note_on", note=note, velocity=VEL, channel=MIDI_CH))


def send_note_off(outport_obj: mido.ports.BaseOutput, note: int) -> None:
    outport_obj.send(mido.Message("note_off", note=note, velocity=0, channel=MIDI_CH))


def main() -> None:
    # --- MIDI ---
    print("MIDI outputs:", mido.get_output_names())
    outport = mido.open_output(PORT_NAME)

    # --- Load model ---
    model_path = os.path.join(os.path.dirname(__file__), "gesture_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # --- Camera ---
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow = faster on Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        outport.close()
        raise RuntimeError("Could not open camera 0")

    cv2.namedWindow("capture image", cv2.WINDOW_NORMAL)

    # --- MediaPipe Hands (Solutions API) ---
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
        # model_complexity=0,  # uncomment for more speed if acceptable
    )

    # --- Gesture stability state ---
    candidate = None
    cand_count = 0
    stable_gesture = None
    none_count = 0

    current_note = None

    last_print = 0.0  # throttle console spam

    try:
        while True:
            success, frame = cap.read()
            if not success:
                # Avoid tight loop if camera hiccups
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            # Determine "active_gesture" for RIGHT hand only (this frame)
            active_gesture = None

            if result.multi_hand_landmarks and result.multi_handedness:
                h, w = frame.shape[:2]

                for hand_landmarks, handed in zip(result.multi_hand_landmarks, result.multi_handedness):
                    hand_label = handed.classification[0].label  # 'Right' or 'Left'
                    if hand_label != "Right":
                        # Still draw if you want, but ignore for control
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        continue

                    coords = []
                    for lm in hand_landmarks.landmark:
                        coords.extend([lm.x, lm.y, lm.z])

                    # Predict gesture (expects list-of-features)
                    gesture = model.predict([coords])[0]
                    active_gesture = str(gesture)

                    wrist = hand_landmarks.landmark[0]
                    px, py = int(wrist.x * w), int(wrist.y * h)
                    cv2.putText(
                        frame,
                        f"{hand_label}: {active_gesture}",
                        (px, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Only use the first Right hand found
                    break

            # --- Stability / debouncing ---
            if active_gesture == candidate:
                cand_count += 1
            else:
                candidate = active_gesture
                cand_count = 1

            if candidate is None:
                none_count += 1
                if none_count >= STABLE_OFF:
                    stable_gesture = None
            else:
                none_count = 0
                if cand_count >= STABLE_ON:
                    stable_gesture = candidate

            # --- MIDI state machine (ONLY ONE place where we send notes) ---
            target_note = GESTURE_TO_NOTE.get(stable_gesture)

            if target_note != current_note:
                if current_note is not None:
                    send_note_off(outport, current_note)

                if target_note is not None:
                    send_note_on(outport, target_note)

                current_note = target_note

            # Throttle prints so VS Code doesn't crawl
            now = time.time()
            if now - last_print > 0.25:
                if stable_gesture is None:
                    print("Stable: None")
                else:
                    print(f"Stable: {stable_gesture} -> {current_note}")
                last_print = now

            cv2.imshow("capture image", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # allow window close button to exit
            if cv2.getWindowProperty("capture image", cv2.WND_PROP_VISIBLE) < 1:
                break

            # tiny sleep to keep CPU sane
            time.sleep(0.001)

    finally:
        # Clean shutdown
        try:
            if current_note is not None:
                send_note_off(outport, current_note)
            midi_panic(outport)
            outport.close()
        except Exception as e:
            print("MIDI cleanup error:", e)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()