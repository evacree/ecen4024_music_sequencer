# QTGUI/src/main_TEST.py
# Evan Acree | 4/13/26
import sys
import cv2
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QLabel, QSpinBox, QGroupBox, QScrollArea,
    QSlider
)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
import os
import mido
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

GESTURES = ["ok", "fist", "one", "two", "three", "four", "five", "six"]

GESTURE_TO_NOTE = {
    "ok": 60, 
    "fist" : 62,         #  major scale   # 
    "one" : 64,
    "two" : 65,
    "three" : 67,
    "four" : 69,
    "five" : 71,
    "six" : 72 #end
}

MIDI_TO_NOTE = {
    60: "C4",
    62: "D4",
    64: "E4",
    65: "F4",
    67: "G4",
    69: "A4",
    71: "B4",
    72: "C5"
}


class CameraWorker(QThread):
    frame_ready = pyqtSignal(QImage)
    gesture_ready = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = True
        self.cap = None

    def run(self):
        try:
            # Load your model
            MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "MediaPipe", "gesture_model.pkl")
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)   # load MediaPipe model.

            import mediapipe as mp
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                                   min_detection_confidence=0.5, min_tracking_confidence=0.5)
            mp_draw = mp.solutions.drawing_utils

            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            while self.running and self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                gesture = None
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        # Get gesture from MediaPipe model
                        coords = []
                        for lm in hand_landmarks.landmark:
                            coords.extend([lm.x, lm.y, lm.z])
                        gesture = data.predict([coords])[0]

                # Send frame to GUI
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
                self.frame_ready.emit(qt_image)

                # Send gesture to sequencer
                if gesture:
                    self.gesture_ready.emit(gesture)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            if self.cap:
                self.cap.release()

    def stop(self):
        self.running = False
        self.wait()


def launch_gui(sequencer):
    app = QApplication(sys.argv)
    window = SequencerGUI(sequencer)
    window.show()
    sys.exit(app.exec())


class SequencerGUI(QMainWindow):
    def __init__(self, sequencer):
        super().__init__()
        self.sequencer = sequencer
        self.step_buttons = {}
        self.visual_offset = -0.7

        self.setWindowTitle("Hand Gesture Music Sequencer")
        self.setGeometry(100, 100, 1550, 820)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # LEFT PANEL
        left_panel = QVBoxLayout()
        main_layout.addLayout(left_panel, stretch=2)

        controls = QHBoxLayout()
        left_panel.addLayout(controls)

        bpm_group = QGroupBox("BPM")
        bpm_layout = QHBoxLayout()
        self.bpm_spin = QSpinBox()
        self.bpm_spin.setRange(20, 300)
        self.bpm_spin.setValue(int(self.sequencer.bpm))
        self.bpm_spin.setMinimumWidth(90)
        self.bpm_spin.valueChanged.connect(self.change_bpm)
        bpm_layout.addWidget(QLabel("BPM:"))
        bpm_layout.addWidget(self.bpm_spin)
        bpm_group.setLayout(bpm_layout)
        controls.addWidget(bpm_group)

        offset_group = QGroupBox("Visual Offset")
        offset_layout = QHBoxLayout()
        self.offset_slider = QSlider(Qt.Orientation.Horizontal)
        self.offset_slider.setRange(-40, 40)
        self.offset_slider.setValue(int(self.visual_offset * 10))
        self.offset_slider.valueChanged.connect(self.change_offset)
        self.offset_label = QLabel(f"Offset: {self.visual_offset:.1f}")
        offset_layout.addWidget(QLabel("Offset:"))
        offset_layout.addWidget(self.offset_slider)
        offset_layout.addWidget(self.offset_label)
        offset_group.setLayout(offset_layout)
        controls.addWidget(offset_group)

        self.start_btn = QPushButton("▶ Start")
        self.stop_btn = QPushButton("⏹ Stop")
        self.reset_btn = QPushButton("Reset All")
        self.start_btn.clicked.connect(self.start_sequencer)
        self.stop_btn.clicked.connect(self.stop_sequencer)
        self.reset_btn.clicked.connect(self.reset_sequencer)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)
        controls.addWidget(self.reset_btn)

        self.status_label = QLabel("Step: 0 | Measure: 1")
        self.status_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        left_panel.addWidget(self.status_label)

        grid_group = QGroupBox("Sequencer Grid - Click or gesture to toggle notes")
        grid_v = QVBoxLayout()
        grid_group.setLayout(grid_v)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        grid_v.addWidget(scroll)
        grid_widget = QWidget()
        self.grid = QGridLayout(grid_widget)
        self.grid.setSpacing(4)
        scroll.setWidget(grid_widget)
        left_panel.addWidget(grid_group)

        # RIGHT PANEL - CAMERA
        right_panel = QVBoxLayout()
        main_layout.addLayout(right_panel, stretch=1)

        camera_group = QGroupBox("Live Hand Tracking")
        camera_layout = QVBoxLayout()
        camera_group.setLayout(camera_layout)

        self.videoLabel = QLabel()
        self.videoLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.videoLabel.setMinimumSize(640, 480)
        self.videoLabel.setStyleSheet("background-color: black; border: 1px solid #555;")
        camera_layout.addWidget(self.videoLabel)
        right_panel.addWidget(camera_group)

        # Build UI
        self.build_grid()
        self.sequencer.step_signal.step_changed.connect(self.on_step_changed)
        self.sequencer.step_signal.sequence_changed.connect(self.refresh_grid)

        self.refresh_grid()
        self.on_step_changed(0)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_status)
        self.timer.start(80)

        # Start camera
        self._init_camera_thread()

    def build_grid(self):
        total_steps = self.sequencer.get_sequence_state()["total_steps"]
        self.grid.addWidget(QLabel("Gesture"), 0, 0)
        for step in range(total_steps):
            lbl = QLabel(str(step))
            if step % 8 == 0:
                lbl.setText(f"{step}\nM{step//8 + 1}")
                lbl.setStyleSheet("font-weight: bold; color: #ddd;")
            self.grid.addWidget(lbl, 0, step + 1)

        for row, gesture in enumerate(GESTURES, 1):
            self.grid.addWidget(QLabel(gesture), row, 0)
            for step in range(total_steps):
                btn = QPushButton()
                btn.setFixedSize(28, 28)
                btn.setCheckable(True)
                btn.clicked.connect(lambda _, g=gesture, s=step: self.toggle_step(g, s))
                self.grid.addWidget(btn, row, step + 1)
                self.step_buttons[(gesture, step)] = btn

    def toggle_step(self, gesture, step):
        note = GESTURE_TO_NOTE.get(gesture)
        if not note:
            return
        notes = self.sequencer.sequence[step][gesture]
        if note in notes:
            notes.remove(note)
            if not notes:
                self.sequencer.track_pitches.pop(gesture, None)
        else:
            notes.append(note)
            self.sequencer.track_pitches[gesture] = note
        self.refresh_grid()

    def change_bpm(self, value):
        self.sequencer.set_bpm(float(value))

    def change_offset(self, value):
        self.visual_offset = value / 10.0
        self.offset_label.setText(f"Offset: {self.visual_offset:.1f}")
        self.on_step_changed(self.sequencer.current_step)

    def start_sequencer(self):
        self.sequencer.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_sequencer(self):
        self.sequencer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def reset_sequencer(self):
        self.sequencer.clear_all()
        self.sequencer.current_step = 0
        self.refresh_grid()
        self.on_step_changed(0)

    def refresh_grid(self):
        for (gesture, step), btn in self.step_buttons.items():
            note = GESTURE_TO_NOTE.get(gesture)
            has_note = note and note in self.sequencer.sequence[step].get(gesture, [])
            btn.setChecked(has_note)
            btn.setStyleSheet("background-color: #22c55e;" if has_note else "")

    def on_step_changed(self, actual_step):
        total_steps = self.sequencer.get_sequence_state()["total_steps"]
        visual_step = (actual_step + self.visual_offset) % total_steps
        measure = (actual_step // 8) + 1
        self.status_label.setText(f"Step: {actual_step} | Measure: {measure} | BPM: {self.sequencer.bpm:.0f}")

        for (gesture, s), btn in self.step_buttons.items():
            if s == round(visual_step):
                btn.setStyleSheet("background-color: #3b82f6; border: 3px solid #ffffff;")
            elif btn.isChecked():
                btn.setStyleSheet("background-color: #22c55e;")
            else:
                btn.setStyleSheet("")

    def update_status(self):
        pass

    def _init_camera_thread(self):
        self.camera_thread = QThread(self)
        self.camera_worker = CameraWorker(camera_index=0)
        self.camera_worker.moveToThread(self.camera_thread)

        self.camera_thread.started.connect(self.camera_worker.start)
        self.camera_worker.frame_ready.connect(self.on_frame_ready)
        self.camera_worker.gesture_ready.connect(self.on_gesture_ready)
        self.camera_worker.error.connect(lambda msg: print("Camera Error:", msg))

        self.camera_thread.start()

    def on_frame_ready(self, qimage):
        pixmap = QPixmap.fromImage(qimage)
        scaled = pixmap.scaled(self.videoLabel.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.videoLabel.setPixmap(scaled)

    def on_gesture_ready(self, gesture):
        if hasattr(self.sequencer, 'handle_gesture'):
            self.sequencer.handle_gesture(gesture)

    def closeEvent(self, event):
        if hasattr(self, "camera_worker"):
            self.camera_worker.stop()
        if hasattr(self, "camera_thread"):
            self.camera_thread.quit()
            self.camera_thread.wait(1000)
        self.sequencer.stop()
        if self.sequencer.midi_out:
            self.sequencer.midi_out.close()
        event.accept()


# ==================== CAMERA WORKER (MediaPipe) ====================
class CameraWorker(QThread):
    frame_ready = pyqtSignal(QImage)
    gesture_ready = pyqtSignal(str)   # Only emitted when gesture is stable.
    error = pyqtSignal(str)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = True

        self.STABLE_ON = 4      # frames needed to confirm a gesture
        self.STABLE_OFF = 4     # frames needed to confirm gesture release

        self.candidate = None
        self.cand_count = 0
        self.stable_gesture = None
        self.none_count = 0

    def run(self):
        try:
            import pickle
            MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "MediaPipe", "gesture_model.pkl")
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)

            import mediapipe as mp
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                                   min_detection_confidence=0.5, min_tracking_confidence=0.5)
            mp_draw = mp.solutions.drawing_utils

            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            while self.running and cap.isOpened():
                success, frame = cap.read()
                if not success:
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                active_gesture = None

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        coords = []
                        for lm in hand_landmarks.landmark:
                            coords.extend([lm.x, lm.y, lm.z])
                        detected = model.predict([coords])[0]

                        midi2note = MIDI_TO_NOTE.get(GESTURE_TO_NOTE.get(detected), "Unknown")
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        wrist = hand_landmarks.landmark[0]

                        text = f"{detected}: {midi2note}"
                        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1, 2)
                        x = frame.shape[1] - text_width - 10
                        y = frame.shape[0] - 10
                        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        # Map detected gesture
                        if detected == "ok" and wrist.x > 0.5:
                            active_gesture = "ok"
                        elif detected == "thumbs_up" and wrist.x > 0.5:
                            active_gesture = "thumbs_up"
                        elif detected == "one" and wrist.x > 0.5:
                            active_gesture = "one"
                        elif detected == "two" and wrist.x > 0.5:
                            active_gesture = "two"
                        elif detected == "three" and wrist.x > 0.5:
                            active_gesture = "three"
                        elif detected == "four" and wrist.x > 0.5:
                            active_gesture = "four"
                        elif detected == "five" and wrist.x > 0.5:
                            active_gesture = "five"
                        elif detected == "six" and wrist.x > 0.5:
                            active_gesture = "six"
                        elif detected == "fist" and wrist.x > 0.5   :
                            active_gesture = "fist"


                # === STABILITY FILTER (prevents spam) ===
                if active_gesture == self.candidate:
                    self.cand_count += 1
                else:
                    self.candidate = active_gesture
                    self.cand_count = 1

                if self.candidate is None:
                    self.none_count += 1
                    if self.none_count >= self.STABLE_OFF:
                        self.stable_gesture = None
                else:
                    self.none_count = 0
                    if self.cand_count >= self.STABLE_ON:
                        if self.stable_gesture != self.candidate:
                            self.stable_gesture = self.candidate
                            self.gesture_ready.emit(self.stable_gesture) 

                # Send frame to GUI
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
                self.frame_ready.emit(qt_image)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            if 'cap' in locals():
                cap.release()

    def stop(self):
        self.running = False
        self.wait()