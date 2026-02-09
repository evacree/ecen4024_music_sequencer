import argparse
import csv
import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

# Arguments for data collection and initialization

def parse_input_arguments():
    parser = argparse.ArgumentParser(
        description="Collect hand landmark data for a gesture label.")
    parser.add_argument("--label", required=True, help="Name of the gesture to capture")
    parser.add_argument("--samples", type=int, default=200, help="Number of frames to collect.")
    parser.add_argument("--warmup", type = float, default = 3.0, help = "Seconds to allow for users to setup before the data collection.")
    parser.add_argument("--outfile", default="data/gesture_data.csv", help="CSV Output Path/Location.")    
    parser.add_argument("--model", default=r"C:\Users\Rich\Desktop\Mediapipe and Machine Learning\Workbench Mediapipe\models\hand_landmarker.task", help="Path to MediaPipe hand landmarker .task model file.")
    parser.add_argument("--cooldown", type=float, default=0.3, help="Seconds between captures.")
    return parser.parse_args()

# CSV file setup for joint coordinates

def csv_fileSetup(outfile):
    out_dir = os.path.dirname(outfile)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    file_exists = os.path.isfile(outfile)

    headers = []
    for i in range(21):
        headers +=[f"x{i}", f"y{i}", f"z{i}"]
    headers.append("label")

    if not file_exists:
        with open(outfile, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

  
def main():

    args = parse_input_arguments()
    csv_fileSetup(args.outfile)

    start_time = time.time()
    warmup_end = start_time + args.warmup

    options = HandLandmarkerOptions(
        base_options = BaseOptions(model_asset_path=args.model),
        running_mode = VisionRunningMode.IMAGE, # Per-frame detection 
        num_hands = 2,
    )

    landmarker = HandLandmarker.create_from_options(options)

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("\n\tCould not open webcam.")
  
    collected = 0
    last_capture = 0.0

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        conv_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data = conv_frame_rgb)

        results = landmarker.detect(mp_image)

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                h, w, _ = frame.shape
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 225, 0), -1)

                # Time counter and warmup for data collection
                now = time.time()
                warming_up = now < warmup_end

                # Screen countdown
                if warming_up:
                    remaining = warmup_end - now
                    cv2.putText(frame, f"Training in: {remaining:.1f}s",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


                # When deciding to write a sample and CSV row writing
                # Previous one: if now - last_capture >= args.cooldown and collected < args.samples:
                if (not warming_up) and (now - last_capture >= args.cooldown) and (collected < args.samples):
                    row = []
                    for lm in hand_landmarks:
                        row.extend([lm.x, lm.y, lm.z])
                    row.append(args.label)

                    with open(args.outfile, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(row)
                    
                    collected += 1
                    last_capture = now

        cv2.putText(
            frame,
            f"Label: {args.label} Collected: {collected}/{args.samples}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Collectuing Data (Tasks API) - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if collected >= args.samples:
            break
        
    capture.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()

