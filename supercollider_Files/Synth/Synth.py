import subprocess
import time
from pythonosc.udp_client import SimpleUDPClient

# Path to your SuperCollider file
SC_FILE = "Synth4ChannelVer3.scd"

# Launch SuperCollider (sclang)
# Use path to sclang.exe file
sc_process = subprocess.Popen(["C:\Program Files\SuperCollider-3.14.1\sclang.exe", SC_FILE])

try:
    sc_process.wait()
except KeyboardInterrupt:
    sc_process.terminate()