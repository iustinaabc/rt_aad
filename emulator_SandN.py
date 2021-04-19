#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:50:40 2020

@author: Nele Eeckman, Sofie Mareels
"""

import time
import numpy as np
from random import random as rand
from pylsl import StreamInfo, StreamOutlet
from group_by_class import group_by_class
from scipy.io import loadmat


def emulate(eeg_data):
    # first create a new stream info (here we set the name to BioSemi,
    # the content-type to EEG, 24 channels, 120 Hz, and float-valued data) The
    # last value would be the serial number of the device or some other more or
    # less locally unique identifier for the stream as far as available (you
    # could also omit it but interrupted connections wouldn't auto-recover)
    info = StreamInfo('BioSemi', 'EEG', 24, 120, 'float32', 'myuid34234')

    eeg_data = np.transpose(eeg_data, (0, 2, 1))

    i = 0
    max = np.shape(eeg_data)[0]

    # next make an outlet
    outlet = StreamOutlet(info)

    while True:
        for j in range(7200):
            mysample = np.array(eeg_data)[i][j]  # 24x1
            outlet.push_sample(mysample)
            time.sleep(1/240)
        i += 1
        if i == max:
            # break
            print("ALERT")
            i = 0
