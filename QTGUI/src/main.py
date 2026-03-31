from PySide6.QtWidgets import *
# from PySide6 import QtWidgets, QHBoxLayout
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtCore import Qt, QTimer, QThread, QObject, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Camera worker object to thread it

class CameraWorker(QObject):
    """Worker to execute camera background tasks"""
    frame_ready = Signal(QImage)
    gesture_ready = Signal(str)
    error = Signal(str)
    finished = Signal()

    def __init__(self, camera_index=0, model=None):
        super(). __init__()
        self.camera_index = camera_index
        self.model = model
        self.running = False

    @Slot()
    def process(self):
        self.running = True
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            self.error.emit("\nCould not open camera.\n")
            self.finished.emit()
            return
            
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands

        with mp_hands.Hands(
            static_image_mode = False,
            max_num_hands = 2,
            min_detection_confidence = 0.6,
            min_tracking_confidence = 0.6,
            model_complexity = 0,
        ) as hands:
                
            while self.running:
                success, frame = cap.read()
                if not success:
                    continue

                frame = cv2.flip(frame ,1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                active_gesture = "none"

                if result.multi_hand_landmarks and result.multi_handedness:
                    h, w = frame.shape[:2]
                    for hand_landmarks, handed in zip(result.multi_hand_landmarks, result.multi_handedness):
                        hand_label = handed.classification[0].label

                        if hand_label != "Right":
                            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            continue

                        coords = []
                        for lm in hand_landmarks.landmark:
                            coords.extend([lm.x, lm.y, lm.z])

                        if self.model is not None:
                            gesture = self.model.predict([coords])[0]
                            active_gesture = str(gesture)
                        else:
                            active_gesture = "no-model"

                        wrist = hand_landmarks.landmark[0]
                        px, py = int(wrist.x * w), int(wrist.y * h)

                        cv2.putText(frame, f"{hand_label}: {active_gesture}",
                                    (px, py - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                                    )
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        break

                rgb_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_out.shape
                qimg = QImage(rgb_out.data, w, h, ch * w, QImage.Format_RGB888).copy()

                self.frame_ready.emit(qimg)
                self.gesture_ready.emit(active_gesture)

        cap.release()
        self.finished.emit()

    @Slot()
    def stop(self):
        self.running = False



# Define a window class
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture Sequencer")
        # self.setMinimumSize(500, 240)
        self.resize(920, 480)
        self.model = None # Here will be added the trained model once needed to integrate

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

        # self.init_camera_and_mediapipe()
        # self.model = None

    
        self.thread = QThread()
        self.worker = CameraWorker(camera_index=0, model=self.model)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.process)
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.gesture_ready.connect(self.on_gesture_ready)
        self.worker.error.connect(print)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()


    @Slot(QImage)
    def on_frame_ready(self, qimg):
        pix = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pix)

    @Slot(str)
    def on_gesture_ready(self, gesture):
        print(f"Gesture: {gesture}")
        # pass
    
    @Slot(str)
    def on_error(self, message):
        print(message)

    def closeEvent(self, event):
        if hasattr(self, "worker"):
            self.worker.stop()
        if hasattr(self, "thread"):
            self.thread.quit()
            self.thread.wait()
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

