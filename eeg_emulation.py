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


def emulate(all=True, left=False):
    # first create a new stream info (here we set the name to BioSemi,
    # the content-type to EEG, 24 channels, 120 Hz, and float-valued data) The
    # last value would be the serial number of the device or some other more or
    # less locally unique identifier for the stream as far as available (you
    # could also omit it but interrupted connections wouldn't auto-recover)
    info = StreamInfo('BioSemi', 'EEG', 24, 120, 'float32', 'myuid34234')

    i = 0
    imax = i + 36

    # next make an outlet
    outlet = StreamOutlet(info)
    # DONE: change location eeg_left & right
    data_subject = loadmat('dataSubject8.mat')
    attended_ear = np.squeeze(np.array(data_subject.get('attendedEar')))
    eeg_data = np.squeeze(np.array(data_subject.get('eegTrials')))
    eeg_left, eeg_right = group_by_class(eeg_data, attended_ear)
    eeg_left = np.transpose(eeg_left, (0, 2, 1))
    eeg_right = np.transpose(eeg_right, (0, 2, 1))
    if not all:
        imax = 24
        if left:
            eeg_data = eeg_left
        else:
            eeg_data = eeg_right

    while True:
        for j in range(7200):
            mysample = np.array(eeg_data)[i][j]  # 24x1
            outlet.push_sample(mysample)
            time.sleep(1/240)
        i += 1
        # mag dan waarschijnlijk nog weg:
        if i == imax:
            break
            # i = 0
