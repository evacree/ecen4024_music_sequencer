# QTGUI/src/main_TEST.py
# Evan Acree | 4/1/26
import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QLabel, QSpinBox, QGroupBox, QScrollArea,
    QSlider
)
from PyQt6.QtCore import QTimer, Qt

GESTURES = ["ok", "thumbs_up", "one", "two", "three", "four",
            "five", "six", "seven", "eight", "nine", "ten"]

GESTURE_TO_NOTE = {
    "ok": 60, "thumbs_up": 62, "one": 64, "two": 65, "three": 67,
    "four": 69, "five": 71, "six": 72, "seven": 76, "eight": 78,
    "nine": 80, "ten": 82
}


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
        self.visual_offset = -0.7      # Tuneable on GUI for now.

        self.setWindowTitle("Hand Gesture Music Sequencer")
        self.setGeometry(100, 100, 1350, 780)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Controls.
        controls = QHBoxLayout()
        layout.addLayout(controls)

        # BPM select.
        bpm_group = QGroupBox("BPM")
        bpm_layout = QHBoxLayout()
        self.bpm_spin = QSpinBox()
        self.bpm_spin.setRange(20, 300)
        self.bpm_spin.setValue(int(self.sequencer.bpm))
        self.bpm_spin.valueChanged.connect(self.change_bpm)
        bpm_layout.addWidget(QLabel("BPM:"))
        bpm_layout.addWidget(self.bpm_spin)
        bpm_group.setLayout(bpm_layout)
        controls.addWidget(bpm_group)

        # Visual offset (delay) slider.
        offset_group = QGroupBox("Audio-Visual Offset")
        offset_layout = QHBoxLayout()
        self.offset_slider = QSlider(Qt.Orientation.Horizontal)
        self.offset_slider.setRange(-40, 40)
        self.offset_slider.setValue(int(self.visual_offset * 10))
        self.offset_slider.valueChanged.connect(self.change_offset)
        self.offset_label = QLabel(f"Offset: {self.visual_offset:.1f}")
        offset_layout.addWidget(QLabel("Visual Offset:"))
        offset_layout.addWidget(self.offset_slider)
        offset_layout.addWidget(self.offset_label)
        offset_group.setLayout(offset_layout)
        controls.addWidget(offset_group)

        # Start/stop/reset.
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
        layout.addWidget(self.status_label)

        # Grid (tracks).
        grid_group = QGroupBox("Active Sequencer Display - Click or gesture to toggle notes.")
        grid_v = QVBoxLayout()
        grid_group.setLayout(grid_v)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        grid_v.addWidget(scroll)
        grid_widget = QWidget()
        self.grid = QGridLayout(grid_widget)
        self.grid.setSpacing(4)
        scroll.setWidget(grid_widget)
        layout.addWidget(grid_group)

        self.build_grid()                                      
        self.sequencer.step_signal.step_changed.connect(self.on_step_changed)
        self.sequencer.step_signal.sequence_changed.connect(self.refresh_grid)

        # Initialization of display.
        self.refresh_grid()
        self.on_step_changed(0)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_status)
        self.timer.start(80)

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

        # Visualization (sweeping bar) - single beat only
        for (gesture, s), btn in self.step_buttons.items():
            if s == round(visual_step):
                btn.setStyleSheet("background-color: #3b82f6; border: 3px solid #ffffff;")
            elif btn.isChecked():
                btn.setStyleSheet("background-color: #22c55e;")
            else:
                btn.setStyleSheet("")

    def update_status(self):
        pass

    def closeEvent(self, event):
        self.sequencer.stop()
        if self.sequencer.midi_out:
            self.sequencer.midi_out.close()
        event.accept()