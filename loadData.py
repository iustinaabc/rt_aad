import numpy as np
import os

from scipy.io import loadmat


def loadData(datafile):
    """
    Load the EEG data and labels from the dataset

    Parameters
    ----------
    :param datafile: (str) The name of the datafile to be used
    :param preprocessing: (dict) The preprocessing parameters for the preprocessing of the EEG
    :param varargin: (variable arguments)

    :return: eeg: (3-dimensional numpy array) EEG from the dataset with dimensions channel x time x trial
             attendedEar: (list) List with labels of each trial
             fs: (int) Sampling frequency of the dataset
    """
    if datafile[-4:] == ".mat":
        # load data
        data = loadmat(datafile)
        attendedEar = np.squeeze(np.array(data.get('attendedEar')))
        fs = np.squeeze(data.get('fs'))
        # eegTrials is list of cells (trials with dimension time x channel)
        # Convert to numpy array with dimensions trial(48) x channel(24) x time(7200)

        eegTrials = np.squeeze(np.array(data.get('eegTrials')))
        eeg = np.zeros((eegTrials.shape[0], eegTrials[0].shape[0], eegTrials[0].shape[1]))
        for i in range(eeg.shape[0]):
            eeg[i] = eegTrials[i]
        eeg = np.transpose(eeg, (0, 2, 1))  # [minutes(48), channels(24), time(7200)]

    else:
        arrays = {}
        for filename in os.listdir(datafile):
            if filename.endswith('.npy'):
                arrays[filename] = np.load(filename)
        eeg = arrays["fulleeg"]
        attendedEar = arrays["attendedEar"]
        # fs = arrays["fs"]
        fs = 250
        print(np.shape(eeg))

    return eeg, attendedEar, fs
