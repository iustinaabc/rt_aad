"""
TRAINING SUBJECT INDEPENDENT OR SUBJECT SPECIFIC FBCSP FILTER AND LDA
"""

import numpy as np
import scipy.signal
from loadData import loadData
from trainCSP import trainCSP
from tmprod import tmprod
from segment import segment
import random
from sklearn import covariance
from group_by_class import group_by_class
import matplotlib.pyplot as plt


def logenergy(y):
    outputEnergyVector = np.zeros(len(y))
    for i in range(len(y)):
        outputEnergyVector[i] = sum(j**2 for j in y[i])
    return np.log(outputEnergyVector)


def trainFilters(eeg=None, attendedEar=None, usingData=True, eeg1=None, eeg2=None, fs=120,
                 filterbankBands=np.array([[12], [30]]), timefr=10):
    """
    Can be used both on a dataset to train Subject Independent filters and LDA as on Subject EEG recording to train
    Subject Specific filters and LDA

    Parameters
    ----------
    :param usingData: (bool) True if the FBCSP filter and LDA are trained on a dataset and are Subject Independent
                                False if subject specific EEG data used to train FBCSP filters and LDA.

    :param data: (str) The name of the dataset to train the FBCSP filters and LDA on.
                          --Is only used if usingDataset is True--

    :param eeg: (2-dim numpy array) The EEG recording to train the subject specific FBCSP filters and LDA on.
                                    Dimensions: channels x time
                                    --Is only used if usingDataset is False--

    :param markers: (1-dim numpy array) The markers recorded together with the EEG, marking the attended speaker and thus the class of the
                                        several parts of the EEG recording.
                                        Should have length equal to the amount of trials
                                        --Is only used if usingDataset is False--

    :param trialSize: (int) The amount of samples in one trial
                            Should be equal to floor(len(eeg[i,:])/len(markers))
                            --Is only used if usingDataset is False--

    :param fs: (int) The sampling frequency used to sample the EEG recording (eeg)
                     --Is only used if usingDataset is False--

    :return: CSP (dict): The trained FBCSP filters, containing W, score and traceratio
             coef (list): The trained LDA coefficients
             b (float): The trained LDA bias

    """

    """ SETUP: parameters """
    params = {
        "windowLengths": 5,  # different lengths decision windows to test (in seconds)
        "saveName": "temp",  # name to save results with
        # "conditions": ["-5/5", "quiet"],
        "conditions": ["all"],

        # PREPROCESSING
        "preprocessing": {
            "normalization": True,  # 1:with normalization of regression matrices (column-wise), 0:without normalization
            "subset": 1,  # subset percentage to be taken from training subjects
            "rereference": "None",
            "eegChanSel": []
        },

        # FILTERBANK SETUP
        # "filterbankBands": np.array([[1.2,2,26],[4,2,30]]),  #first row: lower bound, second row: upper bound
        "filterbankBands": filterbankBands,

        # COVARIANCE ESTIMATION
        "cov": {
            "method": "lwcov",  # covariance matrix estimation method: 'classic' / 'lwcov'
            "weighting": False,  # weighting of individual covariance matrices based on distance
            "distance": "riem"  # distance measure to weight training subjects: 'riem' = Riemannian distance / 'KLdiv' = Kullback-Leibner divergence
        },

        # CSP FILTERS
        "csp": {
            "spatial_dim": 6,  # number of CS patterns to retain (in total, per band) (K)
            "optmode": "ratiotrace"  # optimization mode: 'ratiotrace' or 'traceratio'
        }}

    if usingData:  # in case of training on existing data

        # apply FB
        eegTemp = eeg  # (24,7200,48) ---> (7200, 24, 48) ===== np.transpose(onze eeg)
        eeg = np.zeros((eeg.shape[0], len(params["filterbankBands"][0]), eeg.shape[1], eeg.shape[2]), dtype=np.float32)
        for band in range(len(params["filterbankBands"][0])):
            lower, upper = scipy.signal.iirfilter(8, np.array(
                [2 * params["filterbankBands"][0, band] / fs, 2 * params["filterbankBands"][1, band] / fs]))
            eeg[:, band, :, :] = np.transpose(scipy.signal.filtfilt(lower, upper, np.transpose(eegTemp), axis=0))
            # shape eeg: trials (14) x bands (1) x channels (24) x time (7200)
            print(np.shape(eeg[:, band, :, :]))
            mean = np.average(eeg[:, band, :, :], 2)  # shape: trials (14) x channels(24)
            means = np.full((eeg.shape[3], eeg.shape[0], eeg.shape[2]), mean) # channels(24) x trials(14) x time(7200)
            means = np.transpose(means, (1, 2, 0))

            eeg[:, band, :, :] = eeg[:, band, :, :] - means  # trials(14) x (band) x channels(24) x time(7200)
        del eegTemp
        # save results
        X = eeg
        labels = attendedEar

    # TODO: realtime case
    else:  # in case of realtime training

        # apply FB
        all_eeg = [eeg1, eeg2]
        labels = [1]*np.shape(eeg1)[0] + [2]*np.shape(eeg2)[0]
        X = []
        left = True
        for eeg in all_eeg:
            eegTemp = eeg
            eeg = np.zeros((eeg.shape[0], len(params["filterbankBands"][0]), eeg.shape[1], eeg.shape[2]), dtype=np.float32)
            for band in range(len(params["filterbankBands"][0])):
                lower, upper = scipy.signal.iirfilter(8, np.array(
                    [2 * params["filterbankBands"][0, band] / fs, 2 * params["filterbankBands"][1, band] / fs]))
                eeg[:, band, :, :] = np.transpose(scipy.signal.filtfilt(lower, upper, np.transpose(eegTemp), axis=0))
                # shape eeg: trials (14) x bands (1) x channels (24) x time (7200)
                mean = np.average(eeg[:, band, :, :], 2)  # shape: trials (14) x channels(24)
                means = np.full((eeg.shape[3], eeg.shape[0], eeg.shape[2]),
                                mean)  # channels(24) x trials(14) x time(7200)
                means = np.transpose(means, (1, 2, 0))

                eeg[:, band, :, :] = eeg[:, band, :, :] - means  # trials(14) x (band(1)x) channels(24) x time(7200)
            del eegTemp

            if left:
                X = eeg
                left = False
            else:
                X = np.concatenate((X, eeg), axis=0)

    """TRAIN CSP FILTERS"""
    print("---Training CSP---")
    firstBand = True
    CSP = dict()
    for band in range(0, len(params["filterbankBands"][0])):
        Xtrain = X[:, band, :, :]

        # train CSP filter
        [W, score, traceratio] = trainCSP(Xtrain, labels, params["csp"]["spatial_dim"], params["csp"]["optmode"],
                                                   params["cov"]["method"])
        if firstBand:
            CSP["W"] = W[np.newaxis]
            CSP["score"] = [score]
            CSP["traceratio"] = traceratio
            # X: [trials 14, bands 1, channels 24, time 7200]
            # W: [channels 24, spatial dim 6]
            feat = []
            for trial in range(np.shape(X)[0]):
                for window in range(int(60/timefr)):
                    X_feat = np.squeeze(X[trial, band,:,window*timefr*int(fs):(window+1)*timefr*int(fs)])
                    Y = np.dot(np.transpose(CSP["W"][band]), X_feat)
                    feat.append(logenergy(Y))
            firstBand = False
            # Shape Y: [spatial dim 6, time 7200]
            # shape feat: [trials, spatial dim 6 x #bands]
        else:
            W = W[np.newaxis]
            CSP["W"] = np.concatenate((CSP["W"], W), axis=0)
            CSP["score"] = CSP["score"].append(score)
            CSP["traceratio"] = np.concatenate((CSP["traceratio"], traceratio), axis=0)

            # Filter both training and testing data using CSP filters
            # LDA
            feat_temp = []
            for trial in range(np.shape(X)[0]):
                for window in range(int(60/timefr)):
                    X_feat = np.squeeze(X[trial, band, :, window * timefr * int(fs):(window + 1) * timefr * int(fs)])
                    Ytemp = np.dot(np.transpose(CSP["W"][band]), X_feat)
                    feat_temp.append(logenergy(Ytemp))
            feat = np.concatenate((feat, feat_temp), axis=1)
            # shape feat: [trials (#minutes) , spatial dim * nb of bands]

    """CALCULATE THE COEFFICIENTS"""

    f_in_classes = group_by_class(feat, labels, timefr)
    mean1 = np.mean(f_in_classes[0], axis=0)
    mean2 = np.mean(f_in_classes[1], axis=0)
    # # ###plot training feauture###
    # plt.figure()
    # for i in range(np.shape(f_in_classes[0])[0]):
    #     green_scat = plt.scatter(f_in_classes[0][i][0], f_in_classes[0][i][5], color='green', label='Training Class 1')
    # for i in range(np.shape(f_in_classes[1])[0]):
    #     red_scat = plt.scatter(f_in_classes[1][i][0], f_in_classes[1][i][5], color='red', label='Training Class 2')
    # # plt.legend() #("Class 1", "Class 2"))
    # plt.legend(handles=[green_scat, red_scat])
    # plt.title("Training feature vectors of 1st and 6th dimension plotted in 2D")
    # plt.show()
    # plt.close()

    if params["cov"]["method"] == 'classic':
        S = np.cov(np.transpose(feat))
    elif params["cov"]["method"] == 'lwcov':
        S = covariance.ledoit_wolf(feat)[0]

    diff_mean = np.subtract(mean2, mean1)
    sum_mean = np.add(mean1, mean2)
    coef = np.transpose(np.dot(np.linalg.inv(S), diff_mean))
    b = -0.5 * np.dot(coef, sum_mean)

    return CSP, coef, b, f_in_classes
