import numpy as np
from scipy.io import loadmat


def loadData(datafile, preprocessing, varargin):
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

    # load data
    data = loadmat(datafile)
    attendedEar = np.squeeze(np.array(data.get('attendedEar')))
    fs = data.get('fs')

    # if varargin is None:
    #     conditions = 'all'
    # else:
    #     conditions = varargin[0]
    #
    # if conditions == "dry":
    #     indices = np.where(cond == 0)
    #     data["eegTrials"] = data["eegTrials"][indices]
    #     attendedEar = data["attendedEar"][indices]
    # if conditions == "hrtf":
    #     indices = np.where(cond == 1)
    #     data["eegTrials"] = data["eegTrials"][indices]
    #     attendedEar = data["attendedEar"][indices]

    # eegTrials is list of cells (trials with dimension time x channel)
    # Convert to numpy array with dimensions trial(48) x channel(24) x time(7200)

    eegTrials = np.squeeze(np.array(data.get('eegTrials')))
    eeg = np.zeros((eegTrials.shape[0], eegTrials[0].shape[0], eegTrials[0].shape[1]))
    for i in range(eeg.shape[0]):
        eeg[i] = eegTrials[i]
    eeg = np.transpose(eeg, (0, 2, 1))  # [minutes(48), channels(24), time(7200)]

    return eeg, attendedEar, fs
