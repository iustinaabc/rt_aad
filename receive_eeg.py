import numpy as np
import math
from pylsl import local_clock


def receive_eeg(EEG_inlet, timeframe, overlap=0, datatype=np.float32, channels=24):
    """
    Receives the EEG and markers from LabStreamingLayer and shifts them into the eeg array according to the timeframe
    and overlap. Markers are used to define the class of each EEG trial, when EEG is used for training.

    Parameters
    ----------
    :param timeframe: (int) The amount of EEG samples needed from LSL

    :param overlap: (int) The amount of samples to be included and kept from the previous eeg array


    :param datatype: (datatype) Datatype for the eeg array

    :return: eeg: (np.array) 2-dimensional array of eeg samples filled according to the timeframe and overlap
            stamps:
    """
    # Initialize variables

    eeg = np.zeros((timeframe, channels), dtype=datatype)
    stamps = np.zeros((timeframe), dtype=datatype)

    for i in range(timeframe-overlap):
        sample, timestamps = EEG_inlet.pull_sample()
        sample = np.transpose(sample)  #
        eeg[i] = sample
        stamps[i] = timestamps


    #returning eeg [channels, timeframe] and stamps [timeframe, 1]
    return np.transpose(eeg), np.transpose(stamps)


