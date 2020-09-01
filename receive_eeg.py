from scipy.ndimage.interpolation import shift
import numpy as np
import math
from pylsl import local_clock

def receive_eeg(EEG_inlet, timeframe, eeg=None, stamps=None, overlap=0, datatype=np.float32, channels=24):
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
        eeg = np.zeros((timeframe, channels), dtype=datatype)
    else:
        # eeg samples are added in the time dimension
        eeg = np.transpose(eeg)

    if stamps is None:  # For Synchronization
        stamps = np.zeros((timeframe),dtype=datatype)
    sample = None
    i = 0

    # Pull in until full
    first = True
    while True:
        lastSample = sample
        sample, timestamps = EEG_inlet.pull_sample()
        if first:
            clock = local_clock()
            first = False
            
        if timestamps:
            # Add sample to the array if the samples is new
            if lastSample != sample:
                eeg = np.roll(eeg, -1, axis=0)
                eeg = np.append(eeg[:-1], [sample], axis=0)

                shift(stamps, -1, cval=timestamps)

                i += 1

        # If enough samples are added, eeg and markers are returned
        if i == (timeframe - overlap):
            # Normalization
            mean = np.average(eeg[:,:],1)
            means= np.full((eeg.shape[1],eeg.shape[0]),mean)
            means =np.transpose(means)
            eeg[:,:] = eeg[:,:] - means
            eeg[:,:] = eeg[:,:]/np.linalg.norm(eeg[:,:])*eeg.shape[1]
            break

    return np.transpose(eeg), stamps


