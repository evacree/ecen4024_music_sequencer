# Hand Gesture Music Sequencer

A real-time music sequencer and synthesizer based on hand gesture inputs through a live video feed. Created for OkSTATE ECEN4024 - Capstone Design as a Spring 2026 ECE departmental project under Nate Lannan.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [MIDI Configuration](#midi-configuration)
- [Gesture Mapping](#gesture-mapping)
- [Pure Data Sequencer](#pure-data-sequencer)
- [Project Structure](#project-structure)
- [Known Issues](#known-issues)
- [Future Work](#future-work)
- [Authors](#authors)

---

## Overview

This project is a computer vision based hand gesture music sequencer/synthesizer that allows a user to interact with a sequencer using live hand gestures. The camera captures and recognizes movements and gestures, then passes it to a sequencer to be placed. A synthesizer lastly creates live notes and pitches associated with each gesture. A GUI shows the whole process and allows the user to interact with the sequencer with manual input, as well.

The goal of this project is to create a completely touchless, intuitive, and accurate music interface tha combines computer vision, machine learning, MIDI communication, and music production and sequencing into a single interactive system.

## System Architecture

The system is made of four parts:

Gesture Recognition - A camera captures live video of the user's hand. Frames are processed live using OpenCV. MediaPipe Hand Landmarks task detects hand landmarks (joints, wrist, fingetips) for each frame. Landmark data is passed to a trained classification model to predict gesture.

Synthesizer - Real-time sound output is created using Supercollider. Different instruments are utilized - organ (CH 0 & 1), bass (CH 2), and percussion (CH3).

Sequencer - Gestures are mapped to a synthesizer output and pitch, and stored in a two measure sequencer. Notes are placed on beat, which is subdivided at the eighth note. Simple Python packages such as time, threading, & mido (MIDI) are used to implement this.

GUI - A visual output for sequencer, camera window, and control settings are visible/usable for the user. GUI is implemented using PyQT6.

## Features

-Real time hand gesture recognition
-MediaPipe hand landmark detection
-Machine learning model to classify gestures
-Gesture controlled note placement
-Python based sequencing
-MIDI communication between sequencer and synthesizer
-Custom gesture training pipeline
-Touchless music sequencing

## Hardware Requirements

-Computer capable of running Python, PyQT, Supercollider
-USB Camera or built in computer camera is fine
-Speakers/headphones
-Hand

## Software Requirements

absl-py==2.4.0
attrs==25.4.0
cffi==2.0.0
contourpy==1.3.2
cycler==0.12.1
flatbuffers==25.12.19
fonttools==4.61.1
joblib==1.4.2
kiwisolver==1.4.9
matplotlib==3.10.8
mediapipe==0.10.14                    
mido==1.3.3
ml_dtypes==0.5.4
numpy==1.26.4
opencv-contrib-python==4.8.1.78       
packaging==26.0
pandas==2.3.3
pillow==12.1.0
protobuf==4.25.8
pycparser==3.0
pyparsing==3.3.2
PyQt6==6.7.0
PyQt6-Qt6==6.7.0
PyQt6_sip==13.11.1
python-dateutil==2.9.0.post0
python-rtmidi==1.5.8
pytz==2025.2
scikit-learn==1.5.0                   
scipy==1.13.1
sentencepiece==0.2.1
six==1.17.0
sounddevice==0.5.5
threadpoolctl==3.5.0
tzdata==2025.3

## Installation

1. Clone the repository.

2. Install all dependencies above via pip or another python install manager.

3. Install SuperCollider & LoopMIDI.

## Running the Project

Make sure your two virtual MIDI ports (MediaPipe_PureData and PureData_SuperCollider) are created within LoopMIDI.

Run the sequencer.py script to start everything except sound synthesis. Your camera must be working for this.

Run EITHER Synth.py for auto-start of SuperCollider, OR the Synth4ChannelVer3 file within the same folder directly VIA Supercollider GUI.

You should now hear sound if the sequencer starts and a note is present on the track.

## MIDI Configuration

You must have a virtual MIDI port program such as LoopMIDI running while the project is active, as this is how the sequencer & synthesizer communicate. 

The main MIDI port connection is between sequencer.py in Python and Synth4ChannelVer3.scd in SuperCollider. It sends over pitch, volume, channel, and ON/OFF status for each note, allowing the sequencer to correctly turn on and off specific sounds.

## Gesture Mapping

Each gesture is mapped to a certain note.

ok - C3 - put thumb and forefinger together and put up last three fingers

fist - D3 - fist with fingers facing camera

one - E3 - index finger up with fingers facing camera

two - F3 - index and middle finger up
three - G3 - thumb, index, and middle finger up

four - A3 - thumb, index, middle, and pinky finger up

five - B3 - all five fingers up

six - C4 - thumb and pinky fingers out, like shaka, with fingers facing camera

All gestures can be changed, added samples to, retrained, etc using collect_data.py and train_model.py

## Sequencer

-Plays notes
-Stores notes positions in measures
-Prompting synth sounds
-Responds to gestures
-Playback

## Known Issues

-Gesture misclassification can occur.
-Recognition depends on lighting, angle, distance to camera.
-GUI scaling on non-1440p displays is incorrect.
-Changing channels while sequencer is running can cause notes to never be turned off.

## Future Work

-Add left hand gestures for sequencer settings control without having to click the mouse.
-Improve accuracy - different users, hand sizes, distances, camera angles, lightings.
-Include theremin script somehow/able to shift between normal sequencing mode and thermin mode for live play with a previously sequenced back track made by user.
-More instruments!

## Authors

Evan Acree - Sequencer
Abbie Schlatter - Hand Gesture Recogntion
Nikolas Brouwer - Synthesizer
Ricardo Landeros Aranda - GUI
