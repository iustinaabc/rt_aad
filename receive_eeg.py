import numpy as np
import math
from pylsl import local_clock

def receive_eeg(EEG_inlet, timeframe, eeg=None, stamps=None, overlap=0, datatype=np.float32, channels=24, starttime=None, normframe=None):
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
        eeg = np.zeros((normframe, channels), dtype=datatype)
        eeg = np.zeros((timeframe, channels), dtype=datatype)
    else:
        # eeg samples are added in the time dimension
        eeg = np.transpose(eeg)
        eegfinal = np.zeros((timeframe, channels), dtype=datatype)

    if stamps is None:  # For Synchronization
        stamps = np.zeros((timeframe),dtype=datatype)
    i = 0



    # Check till right sample is available
    sample, timestamps = EEG_inlet.pull_sample()
    while starttime and (timestamps + EEG_inlet.time_correction() - starttime < 0):
        sample, timestamps = EEG_inlet.pull_sample()
        print(timestamps + EEG_inlet.time_correction() - starttime)
        pass

    # Pull in until full
    while True:
        lastSample = sample
        sample, timestamps = EEG_inlet.pull_sample()
            
        if timestamps:
            # Add sample to the array if the samples is new
            if lastSample != sample:
                eeg = np.roll(eeg, -1, axis=0)
                eeg = np.append(eeg[:-1], [sample], axis=0)

                stamps[i] = timestamps

                i += 1

        # If enough samples are added, eeg and markers are returned
        if i == (timeframe - overlap):
            # Normalization
            eeg = eeg.T
            mean = np.average(eeg, axis=1)[:, np.newaxis]
            eeg = eeg - mean
            eeg = eeg/np.std(eeg, axis=1)[:,np.newaxis]
            eeg = eeg #/np.linalg.norm(eeg)*eeg.shape[1]
            eeg = eeg.T
            
            eegfinal[timeframe-normframe:,:] = eeg
            break
        if i in list(range(0,timeframe, normframe)):
            eeg = eeg.T
            mean = np.average(eeg, axis=1)[:, np.newaxis]
            eeg = eeg - mean
            eeg = eeg/np.std(eeg, axis=1)[:,np.newaxis]
            eeg = eeg #/np.linalg.norm(eeg)*eeg.shape[1]
            eeg = eeg.T
            eegfinal[i-normframe:i, :] = eeg
            

    print("test; should be equal to 1 or -1", stamps[0] - stamps[249])
    print(eegfinal.shape)
    return eeg.T, stamps


