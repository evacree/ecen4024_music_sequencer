from PySide6.QtWidgets import *
# from PySide6 import QtWidgets, QHBoxLayout
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Define a window class
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture Sequencer")
        # self.setMinimumSize(500, 240)
        self.resize(920, 480)

        grid = QGridLayout()
        self.buttons = []

        track_names = ["Kick", "Snare", "Hihat", "Clap"]
        steps = 8 # Steps

        for row, name in enumerate(track_names):
            # Label on the left
            label = QLabel(name)
            label.setAlignment(Qt.AlignCenter)
            grid.addWidget(label, row, 0)

            row_buttons = []
            for column in range(steps):
                btn = QPushButton("")
                grid.addWidget(btn, row, column + 1) # +1 because col 0 is the label
                row_buttons.append(btn)
            self.buttons.append(row_buttons)

        sequencer_widget = QWidget()
        sequencer_widget.setLayout(grid)

        # --- Camera Positioning | Mediapipe Display ---

        self.video_label = QLabel("Mediapipe")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(320, 240)

        camera_widget = QWidget()
        vbox = QVBoxLayout(camera_widget)
        vbox.addWidget(self.video_label)

        # --- Main horizontal layout ---
        main_layout = QHBoxLayout()
        main_layout.addWidget(sequencer_widget, 2)
        main_layout.addWidget(camera_widget, 3)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.init_camera_and_mediapipe()
        self.model = None

    def init_camera_and_mediapipe(self, camera_index=0):
        # OpenCV camera
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print("Could not open camera")
            return

        # Existing MediaPipe landmarker setup
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
         # model_complexity=0,  # uncomment for more speed if acceptable
        )
            

        # Timer to process one frame at a time
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)  # calls method below
        self.timer.start(30)  # ms

    def process_one_frame_with_mediapipe(self):
        if not hasattr(self, "cap"):
            return None

        success, frame = self.cap.read()
        if not success:
            return None

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if result.multi_hand_landmarks and result.multi_handedness:
            h, w = frame.shape[:2]

            for hand_landmarks, handed in zip(result.multi_hand_landmarks, result.multi_handedness):
                hand_label = handed.classification[0].label  # 'Right' or 'Left'
                if hand_label != "Right":
                # Still draw if you want, but ignore for control
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    continue

                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])

                # Predict gesture (expects list-of-features)
                if self.model is not None:
                    gesture = self.model.predict([coords])[0]
                    active_gesture = str(gesture)
                else:
                    active_gesture = "no-model"


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
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                return frame

    def update_frame(self):
        """Called by QTimer: process one frame and display it in video_label."""
        image = self.process_one_frame_with_mediapipe()
        if image is None:
            return

        # If your processed image is BGR, convert to RGB before showing
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pix)

    def closeEvent(self, event):
        if hasattr(self, "timer"):
            self.timer.stop()
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        event.accept()



if __name__=="__main__":
    app = QApplication([])
    ui = MainWindow()
    ui.show()
    app.exec()










# Create QApplication

# QMainWindow() or QWidget:
    # Build step grid with QGridLayout QCheckBox or QPushButton
    # Play/Stop Buttons QPushButton
    # Timer QTimer

# Call window.show() and app.exec()

# Bare sequencer uses a 8x4 layout (8 steps by 4 tracks)

