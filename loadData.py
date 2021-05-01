import numpy as np
import os
# from keras.datasets import imdb
from scipy.io import loadmat

# # save np.load
# np_load_old = np.load
#
# # modify the default parameters of np.load
# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
#
# # call load_data with allow_pickle implicitly set to true
# (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
#
# # restore np.load for future normal usage
# np.load = np_load_old


def loadData(datafile, noTraining):
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
    if not noTraining:
        if datafile[-4:] == ".mat":
            # load data
            data = loadmat(datafile)
            attendedEar = np.squeeze(np.array(data.get('attendedEar')))
            fs = int(np.squeeze(data.get('fs')))
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
                    location = datafile + "/" +filename
                    arrays[filename] = np.load(location, allow_pickle=True)
            eeg = arrays["eeg.npy"]
            attendedEar = arrays["attendedEar.npy"]
            fs = int(arrays["fs.npy"])

        return eeg, attendedEar, fs

    else:
        arrays = {}
        for filename in os.listdir(datafile):
            if filename.endswith('.npy'):
                location = datafile + "/" + filename
                arrays[filename] = np.load(location, allow_pickle=True)
        CSP = {'W': arrays["CSP.npy"].item().get("W"), 'score': arrays["CSP.npy"].item().get("score"), 'traceratio': arrays["CSP.npy"].item().get("traceratio")}
        coefficients = arrays["coefficient.npy"]
        b = int(arrays["bias.npy"])
        f_in_classes = arrays["TrainingFeatures.npy"]
        decisionWindow = int(arrays["DecisionWindow.npy"])
        fs = int(arrays["fs.npy"])
        filterbankband = arrays["FilterBankBand.npy"]

        return CSP, coefficients, b, f_in_classes, decisionWindow, fs, filterbankband
