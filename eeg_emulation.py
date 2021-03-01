#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:50:40 2020

@author: Jasper Wouters, Nele Eeckman, Sofie Mareels
"""

import time
import numpy as np
from random import random as rand

from pylsl import StreamInfo, StreamOutlet
from group_by_class import group_by_class
from scipy.io import loadmat


def emulate():
    # first create a new stream info (here we set the name to BioSemi,
    # the content-type to EEG, 24 channels, 120 Hz, and float-valued data) The
    # last value would be the serial number of the device or some other more or
    # less locally unique identifier for the stream as far as available (you
    # could also omit it but interrupted connections wouldn't auto-recover)
    info = StreamInfo('BioSemi', 'EEG', 24, 120, 'float32', 'myuid34234')
    
    i = 0
    # next make an outlet
    outlet = StreamOutlet(info)
    # # case WINDOWS / IOS
    # DONE: change location eeg_left & right
    data_subject = loadmat('dataSubject9.mat')
    attended_ear = np.squeeze(np.array(data_subject.get('attendedEar')))
    eeg_data = np.squeeze(np.array(data_subject.get('eegTrials')))
    eeg_left, eeg_right = group_by_class(eeg_data, attended_ear)

    # # case LINUX
    # eeg_left = np.load('/home/rtaad/Desktop/left_eeg1.npy')
    # eeg_right = np.load('/home/rtaad/Desktop/right_eeg1.npy')

    print("[EEG emulator sending data now]")
    while True:
        for j in range(7200):
            # make a new random 24-channel sample; this is converted into a
            # pylsl.vectorf (the data type that is expected by push_sample)
            # mysample = np.array(eeg_left)[i, :, j]
            mysample = np.array(eeg_data)[i][j]
            # mysample = eeg_left[:, int(i * 750):int((i + 1) * 750)]
            #24x1
            # now send it and wait for a bit
            outlet.push_sample(mysample)
            # time.sleep(0.01)
        i += 1
        #mag dan waarschijnlijk nog weg:
        if i == 48:
            break
