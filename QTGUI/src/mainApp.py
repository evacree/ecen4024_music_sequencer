# mainApp.py - Qt Designer GUI + ALL old features + launch_gui restored
# 1080p-friendly window size (fits your screen perfectly)

from PyQt6 import QtWidgets, QtCore, QtGui
from finalguiRescalable import Ui_MainWindow
from sequencer import GestureSequencer
import cv2
import os
import mido
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

GESTURES = ["ok", "fist", "one", "two", "three", "four", "five", "six"]

GESTURE_TO_NOTE = {
    "ok": 60, "fist": 62, "one": 64, "two": 65,
    "three": 67, "four": 69, "five": 71, "six": 72
}

MIDI_TO_NOTE = {60: "C4", 62: "D4", 64: "E4", 65: "F4",
                67: "G4", 69: "A4", 71: "B4", 72: "C5"}


# ==================== CAMERA WORKER (unchanged) ====================
class CameraWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(QtGui.QImage)
    gesture_ready = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = True
        self.STABLE_ON = 5
        self.STABLE_OFF = 5
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
                if not success: continue

                frame = cv2.flip(frame, 1)

                frame_h, frame_w = frame.shape[:2]
                cv2.line(frame, (frame_w // 2, 0), (frame_w // 2, frame_h), (0, 0, 255), 3)

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
                        wrist = hand_landmarks.landmark[0]

                        text = f"{detected}: {midi2note}"
                        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1, 2)
                        x = frame.shape[1] - text_width - 10
                        y = frame.shape[0] - 10
                        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        # Map detected gesture
                        if detected == "ok" and wrist.x > 0.5:
                            active_gesture = "ok"
                        elif detected == "fist" and wrist.x > 0.5:
                            active_gesture = "fist"
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


                # Stability filter
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

                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                qt_image = QtGui.QImage(rgb_image.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)
                self.frame_ready.emit(qt_image)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            if 'cap' in locals():
                cap.release()

    def stop(self):
        self.running = False
        self.wait()


# ==================== LAUNCH HELPER (required by sequencer.py) ====================
def launch_gui(sequencer):
    """Simple launcher called by sequencer.py - keeps everything compatible"""
    app = QtWidgets.QApplication([])
    win = MainWindow(sequencer)
    win.show()
    app.exec()


# ==================== MAIN WINDOW ====================
class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, sequencer):
        super().__init__()
        self.setupUi(self)

        # === 1080p-FRIENDLY WINDOW SIZE (fixed for your screen) ===
        self.resize(1220, 720)           # Smaller default that fits perfectly
        self.setMinimumSize(1050, 620)   # Allows resizing smaller without breaking

        self.sequencer = sequencer
        self.step_buttons = {}
        self.visual_offset = -0.7

        # Connect sequencer signals
        self.sequencer.step_signal.step_changed.connect(self.on_step_changed)
        self.sequencer.step_signal.sequence_changed.connect(self.on_sequence_changed)

        # Buttons
        self.start_btn.clicked.connect(self.sequencer.start)
        self.stop_btn.clicked.connect(self.sequencer.stop)
        self.reset_btn.clicked.connect(self.reset_sequencer)

        # Presets (all 4)
        self.preset1_btn.clicked.connect(lambda: self.send_midi(34))
        self.preset2_btn.clicked.connect(lambda: self.send_midi(35))
        self.preset3_btn.clicked.connect(lambda: self.send_midi(32))
        self.preset4_btn.clicked.connect(lambda: self.send_midi(33))

        # BPM + Offset
        self.bpmSpinBox.setRange(20, 300)
        self.bpmSpinBox.setValue(int(self.sequencer.bpm))
        self.bpmSpinBox.valueChanged.connect(self.on_bpm_changed)

        self.offset_horizontal_slider.setRange(-40, 40)
        self.offset_horizontal_slider.setValue(int(self.visual_offset * 10))
        self.offset_horizontal_slider.valueChanged.connect(self.on_offset_changed)

        # Drawbars + readouts
        self.drawbar_sliders = [self.sub_fundamental, self.sub_third, self.fundamental,
                                self.second_harmonic, self.third_harmonic, self.fourth_harmonic,
                                self.fifth_harmonic, self.sixth_harmonic, self.eight_harmonic]
        self.drawbar_readouts = [self.sub_fundamental_readout, self.sub_third_readout,
                                 self.fundamental_readout, self.second_harmonic_readout,
                                 self.third_harmonic_readout, self.fourth_harmonic_readout,
                                 self.fifth_harmonic_readout, self.sixth_harmonic_readout,
                                 self.eighth_harmonic_readout]

        initial_drawbar_values = [8, 8, 8, 4, 0, 0, 0, 0, 4]

        for i, slider in enumerate(self.drawbar_sliders):
            slider.setRange(0, 8)
            slider.setValue(initial_drawbar_values[i])          
            slider.valueChanged.connect(lambda val, idx=i: self.on_drawbar_changed(idx, val))

        # Pedal, Bass, Channel groups
        self.pedal_group = QtWidgets.QButtonGroup(self)
        self.pedal_group.setExclusive(True)
        self.pedal_group.addButton(self.none_btn)
        self.pedal_group.addButton(self.volume_btn)
        self.pedal_group.addButton(self.wah_btn)
        self.none_btn.clicked.connect(lambda: self.send_midi(24, channel=1))
        self.volume_btn.clicked.connect(lambda: self.send_midi(25, channel=1))
        self.wah_btn.clicked.connect(lambda: self.send_midi(26, channel=1))
        self.none_btn.setChecked(True)

        self.bass_group = QtWidgets.QButtonGroup(self)
        self.bass_group.setExclusive(True)
        self.bass_group.addButton(self.synth_btn)
        self.bass_group.addButton(self.string_btn)
        self.synth_btn.clicked.connect(lambda: self.send_midi(97, channel=2))
        self.string_btn.clicked.connect(lambda: self.send_midi(98, channel=2))
        self.synth_btn.setChecked(True)

        self.channel_group = QtWidgets.QButtonGroup(self)
        self.channel_group.setExclusive(True)
        self.channel_group.addButton(self.channel0_select_btn)
        self.channel_group.addButton(self.channel1_select_btn_2)
        self.channel0_select_btn.clicked.connect(lambda: self.set_channel(0))
        self.channel1_select_btn_2.clicked.connect(lambda: self.set_channel(1))
        self.channel0_select_btn.setChecked(True)

        # Sequencer grid
        self._setup_sequencer_grid()

        # Camera
        self._init_camera_thread()

        # Initial UI
        self.on_sequence_changed()
        self.on_step_changed(0)

    def _setup_sequencer_grid(self):
        gesture_list = GESTURES
        total_steps = self.sequencer.get_sequence_state()["total_steps"]
        for row, gesture in enumerate(gesture_list):
            for col in range(total_steps):
                btn_name = f"btn{row}_{col}"
                btn = getattr(self, btn_name, None)
                if btn:
                    btn.setCheckable(True)
                    btn.clicked.connect(lambda _, g=gesture, s=col: self.toggle_step(g, s))
                    self.step_buttons[(gesture, col)] = btn

    def toggle_step(self, gesture, step):
        note = GESTURE_TO_NOTE.get(gesture)
        if not note: return
        notes = self.sequencer.sequence[step][gesture]
        if note in notes:
            notes.remove(note)
            if not notes:
                self.sequencer.track_pitches.pop(gesture, None)
        else:
            notes.append(note)
            self.sequencer.track_pitches[gesture] = note
        self.on_sequence_changed()

    def on_bpm_changed(self, value):
        self.sequencer.set_bpm(float(value))

    def on_offset_changed(self, value):
        self.visual_offset = value / 10.0
        self.offset_num_display.setText(f"{self.visual_offset:.1f}")
        self.on_step_changed(self.sequencer.current_step)

    def on_drawbar_changed(self, idx, val):
        if idx < len(self.drawbar_readouts):
            self.drawbar_readouts[idx].setText(str(val))
        base = 96 + (idx + 1)
        if self.sequencer.midi_out:
            if val > 0:
                self.sequencer.midi_out.send(mido.Message('note_on', note=108, velocity=100, channel=0))
                self.sequencer.midi_out.send(mido.Message('note_on', note=base, velocity=100, channel=0))
                time.sleep(0.02)
                self.sequencer.midi_out.send(mido.Message('note_off', note=base, velocity=0, channel=0))
                self.sequencer.midi_out.send(mido.Message('note_off', note=108, velocity=0, channel=0))
            else:
                self.sequencer.midi_out.send(mido.Message('note_on', note=base, velocity=100, channel=0))
                time.sleep(0.02)
                self.sequencer.midi_out.send(mido.Message('note_off', note=base, velocity=0, channel=0))

    def send_midi(self, note, velocity=100, channel=0):
        if self.sequencer.midi_out:
            self.sequencer.midi_out.send(mido.Message('note_on', note=note, velocity=velocity, channel=channel))
            QtCore.QTimer.singleShot(50, lambda: self.sequencer.midi_out.send(
                mido.Message('note_off', note=note, velocity=0, channel=channel)))

    def set_channel(self, ch):
        self.sequencer.current_midi_channel = ch

    def reset_sequencer(self):
        self.sequencer.clear_all()
        self.sequencer.current_step = 0
        self.on_sequence_changed()
        self.on_step_changed(0)

    def on_step_changed(self, actual_step):
        total_steps = self.sequencer.get_sequence_state()["total_steps"]
        visual_step = (actual_step + self.visual_offset) % total_steps
        for (gesture, s), btn in self.step_buttons.items():
            note = GESTURE_TO_NOTE.get(gesture)
            has_note = note is not None and note in self.sequencer.sequence[s].get(gesture, [])
            btn.setChecked(has_note)
            if s == round(visual_step):
                btn.setStyleSheet("background-color: #3b82f6; border: 3px solid #ffffff;")
            elif has_note:
                btn.setStyleSheet("background-color: #22c55e;")
            else:
                btn.setStyleSheet("")

    def on_sequence_changed(self):
        self.on_step_changed(self.sequencer.current_step)

    def _init_camera_thread(self):
        self.camera_thread = QtCore.QThread(self)
        self.camera_worker = CameraWorker(camera_index=0)
        self.camera_worker.moveToThread(self.camera_thread)
        self.camera_thread.started.connect(self.camera_worker.start)
        self.camera_worker.frame_ready.connect(self.on_frame_ready)
        self.camera_worker.gesture_ready.connect(self.on_gesture_ready)
        self.camera_worker.error.connect(lambda msg: print("Camera Error:", msg))
        self.camera_thread.start()

    def on_frame_ready(self, qimage):
        pixmap = QtGui.QPixmap.fromImage(qimage)
        scaled = pixmap.scaled(self.graphicsView.size(),
                               QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                               QtCore.Qt.TransformationMode.SmoothTransformation)
        if hasattr(self.graphicsView, "setPixmap"):
            self.graphicsView.setPixmap(scaled)
        else:
            scene = QtWidgets.QGraphicsScene(self)
            scene.addPixmap(scaled)
            self.graphicsView.setScene(scene)

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


if __name__ == "__main__":
    seq = GestureSequencer()
    launch_gui(seq)