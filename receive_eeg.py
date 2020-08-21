from scipy.ndimage.interpolation import shift
import numpy as np
import math

def receive_eeg(timeframe, eeg=None, overlap=0, trials=None, datatype=np.float32):
    """
    Receives the EEG and markers from LabStreamingLayer and shifts them into the eeg array according to the timeframe
    and overlap. Markers are used to define the class of each EEG trial, when EEG is used for training.

    Parameters
    ----------
    :param timeframe: (int) The amount of EEG samples needed from LSL

    :param eeg: (numpy array) The previous eeg array to be filled according to the overlap

    :param overlap: (int) The amount of samples to be included and keep from the previous eeg array

    :param trials: (int) The amount of trials with a specific class used in the training of a subject
                         timeframe/trials should be the amount of samples for each trial
                         -- Is None when eeg is not used for training --

    :param datatype: (datatype) Datatype for the eeg array

    :return: eeg: (np.array) 2-dimensional array of eeg samples filled according to the timeframe and overlap
             markers: (np.array) 1-dimensional array to specify the class (attended speaker) of each trial recorded.
    """

    # Initialize variables
    if eeg is None:
        eeg = np.array([0 in range(timeframe)], dtype=datatype)
    if trials is not None:
        markers = np.array([0 in range(trials)], dtype=int)
    sample = None
    i = 0

    # Pull in until full
    while True:
        lastSample = sample
        sample, timestamps = EEG_inlet.pull_sample()
        mark, timestamps = marker_inlet.pull_sample()

        # Add sample to the array if the samples is new
        if lastSample != sample:
            shift(eeg, -1, cval=sample)

            # Every new trial the class is reported in the markers array (+timeframe/(2*trials) samples to make sure
            # the right marker is reported)
            if i in [i*math.ceil(timeframe/trials)+timeframe/(2*trials) for i in range(trials)] and trials is not None:
                shift(markers, -1, cval=mark)
            i += 1

        # If enough samples are added, eeg and markers are returned
        if i == (timeframe - overlap):
            break


    # check?
    return eeg, markers


if __name__ == '__main__':
    lst = np.array([2, 3])
    print(all(lst > 1))
