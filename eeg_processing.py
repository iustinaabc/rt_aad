#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:32:42 2020

@author: Jasper Wouters
"""

import multiprocessing

from pylsl import StreamInlet, resolve_stream
import pylsl.pylsl

import eeg_emulation
# import audio


def main():
    # start emulating EEG, this block is to be removed in the real set-up
    eeg_emulator = multiprocessing.Process(target=eeg_emulation.emulate)
    eeg_emulator.daemon = True # emulator gets killed when main terminates, not sure if necessary
    eeg_emulator.start()
    # when using an external EEG device, consider the following page:
    # https://labstreaminglayer.readthedocs.io/info/time_synchronization.html
    # which is related to time synchronization, for applications where
    # such synchronization matters.

    # start audio test case
    # audio_system = multiprocessing.Process(target=audio.main)
    # # audio_system.daemon = True
    # audio_system.start()

    # resolve an EEG stream on the lab network
    print("looking for an EEG stream... ", end = '')
    streams = resolve_stream('type', 'EEG')
    print("[STREAM FOUND]")

    # create a new inlet to read from the stream
    EEG_inlet = StreamInlet(streams[0])

    # a marker stream on the lab network
    print("looking for a marker stream... ", end='')
    streams = resolve_stream('type', 'Markers')
    print("[STREAM FOUND]")

    # create a new inlet to read from the stream
    marker_inlet = StreamInlet(streams[0])

    try:
        EEG_offset = EEG_inlet.time_correction(timeout=2.0)
        print('EEG offset: ' + str(EEG_offset))
    except pylsl.pylsl.TimeoutError:
        print('EEG offset timed out.')

    try:
        marker_offset = marker_inlet.time_correction(timeout=1.0)
        print('Marker offset: ' + str(marker_offset))
    except pylsl.pylsl.TimeoutError:
        print('Marker offset timed out.')

    last_print = 0
    while True:
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        chunk, timestamps = EEG_inlet.pull_chunk()
        if timestamps:
            # TODO implement processing and gain control here
            # for now we print an EEG chunk every 5 seconds

            timestamp_sec = int(timestamps[0])
            if  timestamp_sec % 5 == 0 and timestamp_sec != last_print:
                print(timestamps, chunk)
                last_print = timestamp_sec

        marker, timestamp = marker_inlet.pull_sample(timeout=0.0)
        if timestamp:
            # this could be used in the future for audio sync purposes
            # however, further work required to investigate the precision
            # of these timestamps
            print(timestamp, marker)

        # when one stream breaks off, this loop stays functional

if __name__ == '__main__':
    main()
