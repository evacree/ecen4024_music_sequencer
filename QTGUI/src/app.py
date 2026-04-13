from PyQt6 import QtCore, QtGui, QtWidgets
from uiconcept import Ui_MainWindow
from main_TEST import GESTURES, GESTURE_TO_NOTE
from main import CameraWorker   # you may rename main.py → camera.py
from PyQt6.QtCore import pyqtSlot, QObject, QTimer, pyqtSignal

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, sequencer):
        super().__init__()
        self.setupUi(self)              # builds the Designer UI [file:3]
        self.sequencer = sequencer      # audio engine from main_TEST.py [file:2]
        self.model = None               # gesture classifier for CameraWorker [file:1]

        self.step_buttons = {}          # (gesture, step) -> QPushButton
        self.visual_offset = -0.7       # matches main_TEST.py behavior [file:2]

        self._init_sequencer_grid()
        self._connect_sequencer_signals()
        self._init_camera_thread()
        self._connect_ui_controls()

    def _init_sequencer_grid(self):
        # Setup button grid
        rows = len(GESTURES)
        steps = 8

        buttons_by_row = []
        for row in range(rows):
            row_buttons = []
            for col in range(steps):
                item = self.gridLayout.itemAtPosition(row, col)
                if item is None:
                    continue
                btn = item.widget()
                row_buttons.append(btn)
            buttons_by_row.append(row_buttons)

            # Assign gesture mapping and connect clicks

        for row, gesture in enumerate(GESTURES):
            for step, btn in enumerate(buttons_by_row[row]):
                btn.setCheckable(True)
                btn.clicked.connect(
                    lambda _, g=gesture, s=step: self.toggle_step(g, s)
                )
                self.step_buttons[(gesture, step)] = btn

                self.refresh_grid()

        # Reuse toggle_step, refresh_grid, and on_step_changes

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

    def refresh_grid(self):
        for (gesture, step), btn in self.step_buttons.items():
            note = GESTURE_TO_NOTE.get(gesture)
            has_note = note and note in self.sequencer.sequence[step].get(gesture, [])
            btn.setChecked(bool(has_note))
            btn.setStyleSheet(
                "background-color: #22c55e;" if has_note else ""
            )

    @pyqtSlot(int) # Play visual animation
    def on_step_changed(self, actual_step): # Moving blue play-ahead
        visual_step = (actual_step + self.visual_offset) % len(self.sequencer.sequence)
        measure = (actual_step // 8) + 1

        self.statusbar.showMessage(f"Step: {actual_step} | Measure: {measure} | BPM: {self.sequencer.bpm:.0f}")

        for (gesture, s), btn in self.step_buttons.items():
            if s == round(visual_step):
                btn.setStyleSheet("background-color: #3b82f6; border: 3px solid #ffffff;")
            elif btn.isChecked():
                btn.setStyleSheet("background-color: #22c55e;")
            else:
                btn.setStyleSheet("")
            
    def _connect_sequencer_signals(self):
        self.sequencer.step_signal.step_changed.connect(self.on_step_changed)
        self.on_step_changed(0)

    def _connect_ui_controls(self):
        # BPM spinbox drives sequencer BPM
        self.spinBox.setRange(20, 300)
        self.spinBox.setValue(int(self.sequencer.bpm))
        self.spinBox.valueChanged.connect(
            lambda v: self.sequencer.set_bpm(float(v))
        )

        self.pushButton_33.clicked.connect(self._toggle_start_stop)
                
        # Track checkboxes can act as mutes
        self.checkBox.toggled.connect(lambda state: self._set_track_enabled(0, state))
        self.checkBox_2.toggled.connect(lambda state: self._set_track_enabled(1, state))
        self.checkBox_3.toggled.connect(lambda state: self._set_track_enabled(2, state))
        self.checkBox_4.toggled.connect(lambda state: self._set_track_enabled(3, state))

    def _toggle_start_stop(self):
        if self.sequencer.running:
            self.sequencer.stop()
            self.pushButton_33.setText("Start")
        else:
            self.sequencer.start()
            self.pushButton_33.setText("Stop")

    def _set_track_enabled(self, track_index, enabled):
        self.sequencer.set_track_enabled(track_index, enabled)

    def _init_camera_thread(self):
        # Add a videoLabel in Designer

        self.videoLabel = QtWidgets.QLabel(self.graphicsView)
        self.videoLabel.setGeometry(self.graphicsView.rect())
        self.videoLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.thread = QtCore.QThread(self)
        self.worker = CameraWorker(camera_index=0, model=self.model)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.process)
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.gesture_ready.connect(self.on_gesture_ready)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.thread.deleteLater)

        self.thread.start()

    @pyqtSlot(QtGui.QImage)
    def on_frame_ready(self, qimg):
        pix = QtGui.QPixmap.fromImage(qimg)
        self.videoLabel.setPixmap(pix)

    @pyqtSlot(str)
    def on_gesture_ready(self, gesture):
        print(f"Gesture: {gesture}")

    @pyqtSlot(str)
    def on_error(self, message):
        print(message)

    def closeEvent(self, event):
        if hasattr(self, "worker"):
            self.worker.stop()
        if hasattr(self, "thread"):
            self.thread.quit()
            self.thread.wait()
        self.sequencer.stop()
        if getattr(self.sequencer, "midi_out", None):
            self.sequencer.midi_out.close()
        event.accept()


class StepSignal(QObject):
    step_changed = pyqtSignal(int)

class DummySequencer:
    def __init__(self, steps=32):
        super().__init__()
        self.bpm = 120.0
        self.steps = steps
        self.sequence = [{g: [] for g in GESTURES} for _ in range(8)]
        self.track_pitches = {}
        self.current_step = 0

        from PyQt6.QtCore import QObject, pyqtSignal

        self.step_signal = StepSignal()
        self.running = False

        self._timer = QTimer()
        self._timer.timeout.connect(self._advance_step)
        self._update_timer_interval()

        self.midi_out = None

    def set_bpm(self, bpm: float):
        self.bpm = max(1.0, float(bpm))
        self._update_timer_interval()

    def start(self):
        if self.running:
            return
        self.running = True
        self._timer.start()

    def stop(self):
        if not self.running:
            return
        self.running = False
        self._timer.stop()

    def set_track_enabled(self, track_index, enabled: bool):
        pass

    def clear_all(self):
        for step in self.sequence:
            for g in step:
                step[g].clear()
        self.track_pitches.clear()
        self.current_step = 0
        self.step_signal.step_changed.emit(self.current_step)

    def _update_timer_interval(self):
        ms_per_step = (60_000 / self.bpm) / 4.0
        self._timer.setInterval(int(ms_per_step))

    def _advance_step(self):
        if not self.running:
            return
        self.current_step = (self.current_step + 1) % self.steps


        self.step_signal.step_changed.emit(self.current_step)

from main_TEST import SequencerGUI

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)

    sequencer = DummySequencer(steps=32)
    window = MainWindow(sequencer)
    window.show()

    sys.exit(app.exec())
