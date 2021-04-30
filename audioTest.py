import wave
import os
import alsaaudio as audio
from audio import LRBalancer, AudioPlayer

import os
import time
import multiprocessing
import numpy as np

import wave
import alsaaudio as audio

def main():
    device_name = 'sysdefault'
    control_name = 'Headphone'
    cardindex = 0 #"PCH"

    volumeThreshold = 25  # in percentage
    volLeft = 20  # Starting volume in percentage
    volRight = 20  # Starting volume in percentage

    wav_fn = os.path.join(os.path.expanduser('~/Desktop'), 'Pilot_1.wav')
    wav_fn1 = os.path.join(os.path.expanduser('~/Desktop'), 'travel.wav')
    # Audio device_name = 'sysdefault'
    control_name = 'Headphone'
    cardindex = 0 #"PCH"

    # Audio Control
    lr_bal = LRBalancer()
    lr_bal.set_control(control_name, device_name, cardindex)

    lr_bal.set_volume_left(volLeft)
    lr_bal.set_volume_right(volRight)
    lr_bal = LRBalancer()
    lr_bal.set_control(control_name, device_name, cardindex)

    lr_bal.set_volume_left(volLeft)
    lr_bal.set_volume_right(volRight)
    ap = AudioPlayer()
    ap.set_device(device_name, cardindex)
    ap.init_play(wav_fn)

    ap.play()
    time.sleep(5)
    ap.pause(True)
    time.sleep(3)
    ap.pause(False)
    time.sleep(3)
    ap.pause(True)
    time.sleep(3)
    ap.pause(False)
    time.sleep(3)
    ap.stop()

    ap = AudioPlayer()

    ap.set_device(device_name, cardindex)
    ap.init_play(wav_fn1)

    ap.play()
    time.sleep(5)
    ap.pause(True)
    time.sleep(3)
    ap.pause(False)
    time.sleep(5)
    ap.pause(True)
    time.sleep(3)
    ap.pause(False)
    time.sleep(5)
    ap.stop()

main()
