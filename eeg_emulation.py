#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:50:40 2020

@author: Jasper Wouters
"""

import time
from random import random as rand

from pylsl import StreamInfo, StreamOutlet


def emulate():
    # first create a new stream info (here we set the name to BioSemi,
    # the content-type to EEG, 21 channels, 100 Hz, and float-valued data) The
    # last value would be the serial number of the device or some other more or
    # less locally unique identifier for the stream as far as available (you
    # could also omit it but interrupted connections wouldn't auto-recover)
    info = StreamInfo('BioSemi', 'EEG', 64, 100, 'float32', 'myuid34234')

    # next make an outlet
    outlet = StreamOutlet(info)

    print("[EEG emulator sending data now]")
    while True:
        # make a new random 8-channel sample; this is converted into a
        # pylsl.vectorf (the data type that is expected by push_sample)
        mysample = [rand() for i in range(64)]
        # now send it and wait for a bit
        outlet.push_sample(mysample)
        time.sleep(0.01)
