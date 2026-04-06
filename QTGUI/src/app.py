from PyQt6 import QtCore, QtGui, QtWidgets
from uiconcept import Ui_MainWindow
from main_TEST import GESTURES, GESTURE_TO_NOTE
from main import CameraWorker   # you may rename main.py → camera.py

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