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


def trainFilters(usingDataset=True, dataset="das-2016", eeg=None, markers=None, trialSize=None, fs=64):
    """
    Can be used both on a dataset to train Subject Independent filters and LDA as on Subject EEG recording to train
    Subject Specific filters and LDA

    Parameters
    ----------
    :param usingDataset: (bool) True if the FBCSP filter and LDA are trained on a dataset and are Subject Independent
                                False if subject specific EEG data used to train FBCSP filters and LDA.

    :param dataset: (str) The name of the dataset to train the FBCSP filters and LDA on.
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
        "windowLengths": np.array([60]),  # different lengths decision windows to test (in seconds)
        "saveName": "temp",  # name to save results with
        # "conditions": ["-5/5", "quiet"],
        "conditions": ["all"],

        # PREPROCESSING
        "preprocessing": {
            "normalization": True,  # 1:with normalization of regression matrices (column-wise), 0:without normalization
            "subset": 0.3,  # subset percentage to be taken from training subjects
            "rereference": "None",
            "eegChanSel": []
        },

        #FILTERBANK SETUP
        # "filterbankBands": np.array([[1,2:2:26],[4:2:30]]),  #first row: lower bound, second row: upper bound
        "filterbankBands": np.array([[14],[26]]),

        #COVARIANCE ESTIMATION
        "cov": {
            "method": "lwcov",  # covariance matrix estimation method: 'classic' / 'lwcov'
            "weighting": False,  # weighting of individual covariance matrices based on distance
            "distance": "riem"  # distance measure to weight training subjects: 'riem' = Riemannian distance / 'KLdiv' = Kullback-Leibner divergence
        },

        #CSP FILTERS
        "csp":{
            "npat": 6,  # number of CS patterns to retain (in total, per band) (K)
            "optmode": "ratiotrace"  # optimization mode: 'ratiotrace' or 'traceratio'
        }}

    if usingDataset:  # When the CSP filters and LDA is trained on a dataset (4-dimensional) and Subject Independent
        if dataset == "das-2016":
            # Inconsistente subject verwijderd (nr 7 (6+1))
            trainingSet = set(range(1,11))-{6}


        firstTrainingSubject = True
        print("Loading data other subjects")
        for sb in trainingSet:
            [eeg, attendedEar, fs, trialLength] = loadData(dataset, sb, params["preprocessing"],
                                                              params["conditions"])

            # Random subset in trial dimension
            ind = random.choices(list(range(0, len(attendedEar))),
                                 k=round(params["preprocessing"]["subset"] * len(attendedEar)))
            attendedEar = attendedEar[ind]
            eeg = eeg[:, :, ind]

            # apply FB
            eegTemp = eeg
            eeg = np.zeros((eeg.shape[0], len(params["filterbankBands"][0]), eeg.shape[1], eeg.shape[2]), dtype=np.float32)
            for band in range(len(params["filterbankBands"][0])):
                b, a = scipy.signal.iirfilter(8, np.array(
                    [2 * params["filterbankBands"][0, band] / fs, 2 * params["filterbankBands"][1, band] / fs]))
                eeg[:, band, :, :] = np.transpose(scipy.signal.filtfilt(b, a, np.transpose(eegTemp, (1, 0, 2)), axis=0),
                                                  (1, 0, 2))
                # eeg now has dimensions channels x time x trials
                mean = np.average(eeg[:, band, :, :], 1)
                means = np.full((eeg.shape[2], eeg.shape[0], eeg.shape[3]), mean)
                means = np.transpose(means, (1, 0, 2))

                eeg[:, band, :, :] = eeg[:, band, :, :] - means
            del eegTemp
            # save results
            if firstTrainingSubject:
                X = eeg
                labels = attendedEar
                firstTrainingSubject = False
            else:
                X = np.concatenate((X, eeg), axis=3)
                labels = np.concatenate((labels, attendedEar))


    else:  # When the training data is subject specific and 2-dimensional in channels x time

        # apply FB
        eegTemp = eeg
        eeg = np.zeros((eeg.shape[0], len(params["filterbankBands"][0]), eeg.shape[1], eeg.shape[2]), dtype=np.float32)
        for band in range(len(params["filterbankBands"][0])):
            b, a = scipy.signal.iirfilter(8, np.array(
                [2 * params["filterbankBands"][0, band] / fs, 2 * params["filterbankBands"][1, band] / fs]))
            eeg[:, band, :, :] = np.transpose(scipy.signal.filtfilt(b, a, np.transpose(eegTemp, (1, 0, 2)), axis=0),
                                              (1, 0, 2))
            # eeg now has dimensions channels x time x trials
            mean = np.average(eeg[:, band, :, :], 1)
            means = np.full((eeg.shape[2], eeg.shape[0], eeg.shape[3]), mean)
            means = np.transpose(means, (1, 0, 2))

            eeg[:, band, :, :] = eeg[:, band, :, :] - means
        del eegTemp

        X = eeg
        labels = markers

        # Segment the EEG recording into the separate trials with a specific class in markers
        X = X[:,:,:,np.newaxis]
        X = np.transpose(X, (0, 2, 1, 3))
        X = segment(X,trialSize)


    """TRAIN CSP FILTERS"""
    print("Training CSP")

    first = True
    CSP = dict()
    for band in range(0, len(params["filterbankBands"][0])):
        Xtrain = np.squeeze(X[:, band, :, :])

        # Covariance weighting

        # train CSP filter
        [W, score, traceratio] = trainCSP(Xtrain, labels, params["csp"]["npat"], params["csp"]["optmode"],
                                                   params["cov"]["method"])

        if first:
            CSP["W"] = W
            CSP["score"] = score
            CSP["traceratio"] = traceratio
            Y = tmprod(np.squeeze(X[:, band, :, :]), np.transpose(CSP["W"][:, :]))

            first = False

        else:
            CSP["W"] = np.concatenate((CSP["W"], W), axis=2)
            CSP["score"] = np.concatenate((CSP["score"], score), axis=2)
            CSP["traceratio"] = np.concatenate((CSP["traceratio"], traceratio), axis=2)

            # Filter both training and testing data using CSP filters
            Y = np.concatenate((Y, tmprod(np.squeeze(X[:, band, :, :]), np.transpose(CSP["W"][:, :, band]))), axis=3)

    if len(params["filterbankBands"][0]) == 1:
        Y = Y[:, :, :, np.newaxis]
        CSP["W"] = CSP["W"][:, :, np.newaxis]
    Y = np.transpose(Y, (0, 3, 1, 2))

    """CALCULATE THE COEFFICIENTS"""
    for w in range(0, len(params["windowLengths"])):
        YtrainWindow = segment(Y, params["windowLengths"][w] * fs)
        labelstrainWindow = np.repeat(labels, np.floor(trialLength / (params["windowLengths"][w] * fs)))
        feat = np.log( np.var( YtrainWindow, axis=2 ) )
        feat = np.transpose(np.reshape(feat, (feat.shape[0] * feat.shape[1],
                                                                feat.shape[2])))

        trainindices1 = np.where(labelstrainWindow == 1)
        trainindices2 = np.where(labelstrainWindow == 2)
        mu = np.transpose(np.array([np.average(feat[trainindices1[0], :], axis=0),
                      np.average(feat[trainindices2[0], :], axis=0)]))

        S = np.cov(np.transpose(feat))
        coef = np.linalg.solve(S, (np.subtract(mu[:, 1], mu[:, 0])))
        b = -np.matmul(np.transpose(coef[:, np.newaxis]), np.sum(mu, axis=1)[:, np.newaxis])*1/2
        b = np.squeeze(np.squeeze(b))


    return CSP, coef, b


