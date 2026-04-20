from PyQt6 import QtWidgets, QtCore
from finalgui import Ui_MainWindow
from sequencer import GestureSequencer

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 1) Create the engine
        self.seq = GestureSequencer()

        # 2) Connect engine → GUI (signals)
        self.seq.step_signal.step_changed.connect(self.on_step_changed)
        self.seq.step_signal.sequence_changed.connect(self.on_sequence_changed)

        # 3) Connect GUI → engine (buttons / controls)
        self.start_btn.clicked.connect(self.seq.start)
        self.stop_btn.clicked.connect(self.seq.stop)

        self.bpmSpinBox.setRange(20, 300)
        self.bpmSpinBox.setValue(int(self.seq.bpm))
        self.bpmSpinBox.valueChanged.connect(self.on_bpm_changed)

        # TODO: connect your channel select buttons, preset buttons, sliders, etc.

    def on_bpm_changed(self, value: int):
        self.seq.set_bpm(float(value))

    def on_step_changed(self, step_index: int):
        # Highlight the current step in your UI (e.g. update LEDs, labels, etc.)
        pass

    def on_sequence_changed(self):
        # Redraw the grid of notes for the current channel
        pass

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    win = MainWindow()
    win.show()
    app.exec()