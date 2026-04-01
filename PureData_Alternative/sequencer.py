# PureData_Alternative/sequencer.py
# Evan Acree | 4/1/26
# TODO: Bugs -- delays on the order of -1.5, -0.5, 0.5, 1.5, ... cause note tracker to show on even beats only.
#       Testing -- UNTESTED with MediaPipe and hand gestures.

import time                                     # Used for sleep, perf_counter.
import threading                                # Used for running sequencer independently (in background) of GUI.
import mido                                     # Used for MIDI I/O.
from collections import defaultdict             # Used for creating dictionaries that avoid empty list errors with default values.
import copy                                     # Used for copying sequence data for GUI.
import sys                                      # Used for finding QTGUI file path.        
import os                                       # Used for finding QTGUI file path.
from PyQt6.QtCore import QObject, pyqtSignal    # Used for creating signals sent to the GUI.

# I/O port names used in LoopMIDI for connecting to MediaPipe (input) and SuperCollider (output).
MIDI_IN_BASE_NAME  = "MediaPipe_to_PureData"
MIDI_OUT_BASE_NAME = "PureData_SuperCollider"

# Sequencer parameters, such as BPM, note clipoff, steps/beats per measure, and number of measures.
BPM = 80.0
NOTE_DURATION_FACTOR = 0.95
STEPS_PER_MEASURE = 8
NUM_MEASURES = 4
TOTAL_STEPS = STEPS_PER_MEASURE * NUM_MEASURES

MIDI_CHANNEL = 0     # Channel number, which SuperCollider must use as well.
VELOCITY = 100       # Volume (0-127, not important).


# Mapping from gesture to MIDI note pitch.
GESTURE_TO_NOTE = {
    "ok": 60,        # C3
    "thumbs_up": 62, # D3
    "one": 64,       # E3
    "two": 65,       # F3
    "three": 67,     # G3
    "four": 69,      # A3
    "five": 71,      # B3
    "six": 72,       # C4
    "seven": 76,     # E4
    "eight": 78,     # F#4
    "nine": 80,      # G#4
    "ten": 82        # A#4
}

# Mapping from MIDI note pitch to gesture.
NOTE_TO_GESTURE = {v: k for k, v in GESTURE_TO_NOTE.items()}

# CLASS: Used for communicating with GUI; tells GUI the current step has changed and what that step is.
class StepSignal(QObject):
    step_changed = pyqtSignal(int)

# CLASS: Major sequencer class which contains all parameters & methods.
class GestureSequencer:

    # METHOD: Sets initial conditions for Sequencer object, such as BPM, step count, etc
    def __init__(self):
        self.bpm = BPM                      # Tempo.
        self.running = False                # On/off status.
        self.current_step = 0               # Current step.
        self.sequence = [defaultdict(list) for _ in range(TOTAL_STEPS)] # Sequencer track structure.
        self.track_pitches = {}             
        self.active_notes = {}
        self.midi_out = None
        self.midi_in = None
        self.sequencer_thread = None
        self.midi_listener_thread = None
        self.last_step_time = 0
        self.step_signal = StepSignal()

        self._open_midi_ports()

    def _open_midi_ports(self):
        input_ports = mido.get_input_names()
        output_ports = mido.get_output_names()

        print("Available MIDI Inputs :", input_ports)
        print("Available MIDI Outputs:", output_ports)

        self.midi_in = None
        for port in input_ports:
            if MIDI_IN_BASE_NAME in port:
                try:
                    self.midi_in = mido.open_input(port)
                    print(f"✅ MIDI Input connected → {port}")
                    break
                except Exception as e:
                    print(f"❌ Failed to open input {port}: {e}")

        if not self.midi_in:
            print(f"❌ Could not find any port containing '{MIDI_IN_BASE_NAME}'")

        self.midi_out = None
        for port in output_ports:
            if MIDI_OUT_BASE_NAME in port:
                try:
                    self.midi_out = mido.open_output(port, autoreset=True)
                    print(f"✅ MIDI Output connected → {port}")
                    break
                except Exception as e:
                    print(f"❌ Failed to open output {port}: {e}")

        if not self.midi_out:
            print(f"❌ Could not find any port containing '{MIDI_OUT_BASE_NAME}'")

        # Start MIDI listener if input was found
        if self.midi_in:
            self.midi_listener_thread = threading.Thread(target=self._midi_input_loop, daemon=True)
            self.midi_listener_thread.start()


    def _midi_input_loop(self):
        while True:
            if self.midi_in:
                for msg in self.midi_in.iter_pending():
                    if msg.type == "note_on" and msg.velocity > 0:
                        if msg.note in NOTE_TO_GESTURE:
                            self.handle_gesture(NOTE_TO_GESTURE[msg.note])
            time.sleep(0.001)

    # METHOD: Sets BPM to a new value.
    def set_bpm(self, new_bpm: float):
        self.bpm = max(20, min(300, new_bpm)) # Range of 20-300 BPM.

    def get_note_duration(self):
        return (60.0 / self.bpm / 2) * NOTE_DURATION_FACTOR

    def add_note(self, step: int, track_id: str, midi_note: int):
        if 0 <= step < TOTAL_STEPS:
            self.sequence[step][track_id].append(midi_note)
            if track_id not in self.track_pitches:
                self.track_pitches[track_id] = midi_note

    def clear_track(self, track_id: str):
        for step in self.sequence:
            if track_id in step:
                step[track_id].clear()
        self.track_pitches.pop(track_id, None)

    def clear_all(self):
        self.sequence = [defaultdict(list) for _ in range(TOTAL_STEPS)]
        self.track_pitches.clear()

    def start(self):
        if self.running:
            return
        self.running = True
        self.last_step_time = time.perf_counter()
        self.sequencer_thread = threading.Thread(target=self._sequencer_loop, daemon=True)
        self.sequencer_thread.start()
        print("🎹 Sequencer STARTED")

    def stop(self):
        self.running = False
        if self.sequencer_thread:
            self.sequencer_thread.join(timeout=1.0)
        self._all_notes_off()

    def _all_notes_off(self):
        if not self.midi_out:
            return
        for notes in list(self.active_notes.values()):
            if isinstance(notes, list):
                for n in notes:
                    self.midi_out.send(mido.Message('note_off', note=n, velocity=0, channel=MIDI_CHANNEL))
            else:
                self.midi_out.send(mido.Message('note_off', note=notes, velocity=0, channel=MIDI_CHANNEL))
        self.active_notes.clear()

    def _sequencer_loop(self):
        while self.running:
            step_duration = self.get_note_duration()
            now = time.perf_counter()

            if now - self.last_step_time >= step_duration:
                self._process_step()
                self.current_step = (self.current_step + 1) % TOTAL_STEPS
                self.step_signal.step_changed.emit(self.current_step)
                self.last_step_time = now

            time_to_next = step_duration - (now - self.last_step_time)
            if time_to_next > 0.005:
                time.sleep(time_to_next - 0.003)
            else:
                while (time.perf_counter() - self.last_step_time) < step_duration:
                    pass

    def _process_step(self):
        if not self.midi_out:
            return
        step_data = self.sequence[self.current_step]

        for track_id in list(self.active_notes.keys()):
            if track_id not in step_data or not step_data[track_id]:
                self._note_off_track(track_id)

        for track_id, notes in step_data.items():
            if notes:
                self._note_on_track(track_id, notes)

    def _note_on_track(self, track_id: str, midi_notes: list):
        self._note_off_track(track_id)
        for note in midi_notes:
            self.midi_out.send(mido.Message('note_on', note=note, velocity=VELOCITY, channel=MIDI_CHANNEL))
        self.active_notes[track_id] = midi_notes[:] if len(midi_notes) > 1 else midi_notes[0]

    def _note_off_track(self, track_id: str):
        if track_id in self.active_notes:
            notes = self.active_notes[track_id]
            if isinstance(notes, list):
                for n in notes:
                    self.midi_out.send(mido.Message('note_off', note=n, velocity=0, channel=MIDI_CHANNEL))
            else:
                self.midi_out.send(mido.Message('note_off', note=notes, velocity=0, channel=MIDI_CHANNEL))
            del self.active_notes[track_id]

    def handle_gesture(self, gesture: str):
        if gesture not in GESTURE_TO_NOTE:
            return
        midi_note = GESTURE_TO_NOTE[gesture]
        track_id = gesture
        current_step = self.current_step

        track_notes = self.sequence[current_step][track_id]
        if midi_note in track_notes:
            track_notes.remove(midi_note)
            if not track_notes:
                self.track_pitches.pop(track_id, None)
        else:
            track_notes.append(midi_note)
            self.track_pitches[track_id] = midi_note

    def get_sequence_state(self):
        return {
            "current_step": self.current_step,
            "bpm": self.bpm,
            "sequence": copy.deepcopy(self.sequence),
            "track_pitches": dict(self.track_pitches),
            "running": self.running,
            "total_steps": TOTAL_STEPS,
            "num_measures": NUM_MEASURES
        }


# ----- Launches and sets up GUI from QTGUI/src/main_TEST.py. -----
if __name__ == "__main__":
    seq = GestureSequencer()

    # Test pattern, remove eventually.
    for i in range(0, 8, 2):
        seq.add_note(i, "ok", 60)
    for i in range(1, 8, 2):
        seq.add_note(i, "three", 67)

    print("\nLaunching GUI from QTGUI/src/main_TEST.py ...\n")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'QTGUI', 'src'))
    from main_TEST import launch_gui
    launch_gui(seq)