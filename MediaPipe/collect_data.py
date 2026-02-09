# import argparse
# import csv
# import os
# import time

# import cv2
# import mediapipe as mp

# def main():
#     parser = argparse.ArgumentParser(description="Collect hand landmark data for a gesture label.")
#     parser.add_argument("--label", required=True, help="Name of the gesture class (e.g., thumbs_up)")
#     parser.add_argument("--samples", type=int, default=200, help="Number of frames to capture")
#     parser.add_argument("--outfile", default="data/gesture_data.csv", help="CSV output path")
#     parser.add_argument("--min_detection_confidence", type=float, default=0.7)
#     parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
#     args = parser.parse_args()

#     os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

#     mp_hands = mp.solutions.hands
#     mp_drawing = mp.solutions.drawing_utils

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         raise RuntimeError("Could not open webcam.")

#     headers = []
#     for i in range(21):
#         headers += [f"x{i}", f"y{i}", f"z{i}"]
#     headers += ["label"]

#     # Create file if it doesn't exist
#     file_exists = os.path.isfile(args.outfile)
#     if not file_exists:
#         with open(args.outfile, "w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow(headers)

#     collected = 0
#     cooldown = 0.3  # seconds between captures
#     last_capture = 0.0

#     with mp_hands.Hands(
#         max_num_hands=2,
#         min_detection_confidence=args.min_detection_confidence,
#         min_tracking_confidence=args.min_tracking_confidence,
#     ) as hands:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = hands.process(image_rgb)

#             if results.multi_hand_landmarks:
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     mp_drawing.draw_landmarks(
#                         frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
#                     )

#                     now = time.time()
#                     if now - last_capture >= cooldown and collected < args.samples:
#                         row = []
#                         for lm in hand_landmarks.landmark:
#                             row.extend([lm.x, lm.y, lm.z])
#                         row.append(args.label)
#                         with open(args.outfile, "a", newline="") as f:
#                             writer = csv.writer(f)
#                             writer.writerow(row)
#                         collected += 1
#                         last_capture = now

#             cv2.putText(frame, f"Label: {args.label}  Collected: {collected}/{args.samples}",
#                         (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#             cv2.imshow("Collecting Data - press q to quit", frame)
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break

#     cap.release()
#     cv2.destroyAllWindows()
#     print(f"Finished. Collected {collected} samples for label '{args.label}'.")

# if __name__ == "__main__":
#     main()

import argparse
import csv
import os
import time

import cv2
import mediapipe as mp


def main():
    parser = argparse.ArgumentParser(description="Collect hand landmark data for a gesture label.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--outfile", default="data/gesture_data.csv")

    # Tasks-specific args
    parser.add_argument("--model", default="models/hand_landmarker.task")
    parser.add_argument("--num_hands", type=int, default=2)
    parser.add_argument("--min_hand_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_hand_presence_confidence", type=float, default=0.5)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

    # CSV header
    headers = []
    for i in range(21):
        headers += [f"x{i}", f"y{i}", f"z{i}"]
    headers.append("label")

    if not os.path.isfile(args.outfile):
        with open(args.outfile, "w", newline="") as f:
            csv.writer(f).writerow(headers)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.model),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=args.num_hands,
        min_hand_detection_confidence=args.min_hand_detection_confidence,
        min_hand_presence_confidence=args.min_hand_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    collected = 0
    cooldown_s = 0.3
    last_capture_t = 0.0

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # For VIDEO mode, timestamp just needs to be monotonic
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                # Just draw simple circles/lines using OpenCV (no extra proto imports)
                for hand in result.hand_landmarks:
                    # Example simple drawing: plot joints
                    h, w, _ = frame_bgr.shape
                    pts = []
                    for lm in hand:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        pts.append((cx, cy))
                        cv2.circle(frame_bgr, (cx, cy), 2, (0, 255, 0), -1)

                    # Connect a few keybones (0–5–9–13–17 is a simple “palm” chain)
                    chain = [0, 5, 9, 13, 17]
                    for i in range(len(chain) - 1):
                        p1, p2 = pts[chain[i]], pts[chain[i + 1]]
                        cv2.line(frame_bgr, p1, p2, (0, 255, 0), 2)

                    now = time.time()
                    if (now - last_capture_t) >= cooldown_s and collected < args.samples:
                        row = []
                        for lm in hand:
                            row.extend([lm.x, lm.y, lm.z])
                        row.append(args.label)

                        with open(args.outfile, "a", newline="") as f:
                            csv.writer(f).writerow(row)

                        collected += 1
                        last_capture_t = now

            cv2.putText(
                frame_bgr,
                f"Label: {args.label}  Collected: {collected}/{args.samples}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Collecting Data - press q to quit", frame_bgr)
            if (cv2.waitKey(1) & 0xFF) == ord("q") or collected >= args.samples:
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished. Collected {collected} samples for label '{args.label}'.")


if __name__ == "__main__":
    main()

