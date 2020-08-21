import scipy.io as spio
import numpy as np


def loadData(dataset,number,preprocessing,varargin):
    """
    Load the EEG data and labels from the dataset

    Parameters
    ----------
    :param dataset: (str) The name of the dataset to be used
    :param number: (int) The number of the subject in the dataset
    :param preprocessing: (dict) The preprocessing parameters for the preprocessing of the EEG
    :param varargin: (variable arguments)

    :return: eeg: (3-dimensional numpy array) EEG from the dataset with dimensions channel x time x trial
             attendedEar: (list) List with labels of each trial
             fs: (int) Sampling frequency of the dataset
             trialLength: (int) Amount of samples in one trial
    """
    if dataset == 'das-2016':
        number = number+1
        # load data
        mat = spio.loadmat('data/dataSubject'+str(number)+'.mat')
        attendedEar = mat["attendedEar"][0]
        fs = 64
        trialLength = fs*60

        if varargin is None:
            conditions = 'all'
        else:
            conditions = varargin[0]

        if conditions == "dry":
            indices = np.where(cond == 0)
            mat["eegTrials"] = mat["eegTrials"][indices]
            attendedEar = mat["attendedEar"][indices]
        if conditions == "hrtf":
            indices = np.where(cond == 1)
            mat["eegTrials"] = mat["eegTrials"][indices]
            attendedEar = mat["attendedEar"][indices]

        # eegTrials is list of cells (trials with dimension time x channel)
        # Convert to numpy array with dimensions channel x time x trial

        eegTrials = np.array(mat["eegTrials"])
        eegTrials = np.squeeze(eegTrials)
        eeg = np.zeros((eegTrials.shape[0], eegTrials[0].shape[0], eegTrials[0].shape[1]))
        for i in range(eeg.shape[0]):
            eeg[i] = eegTrials[i]
        eeg = np.transpose(eeg, (2, 1, 0))

        cz = 48

    ntrials = len(attendedEar)
    trialLength = eeg.shape[1]

    # Preprocessing
    if preprocessing["eegChanSel"] != []:
        eeg = eeg[preprocessing["eegChanSel"], :, :]

    if preprocessing["normalization"]:
        for tr in range(ntrials):
            mean = np.average(eeg[:, :, tr], 1)
            means = np.full((eeg.shape[1], eeg.shape[0]), mean)
            means = np.transpose(means)
            eeg[:, :, tr] = eeg[:, :, tr] - means
            eeg[:, :, tr] = eeg[:, :, tr]/np.linalg.norm(eeg[:, :, tr])*eeg.shape[1]

    return eeg, attendedEar, fs, trialLength
