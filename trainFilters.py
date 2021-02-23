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


def trainFilters(dataset, usingDataset=True, eeg=None, markers=None, trialSize=None, fs=250, windowLength=None):
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
        "windowLengths": 5,  # different lengths decision windows to test (in seconds)
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
        "filterbankBands": np.array([[12],[30]]),

        #COVARIANCE ESTIMATION
        "cov": {
            "method": "classic",  # covariance matrix estimation method: 'classic' / 'lwcov'
            "weighting": False,  # weighting of individual covariance matrices based on distance
            "distance": "riem"  # distance measure to weight training subjects: 'riem' = Riemannian distance / 'KLdiv' = Kullback-Leibner divergence
        },

        #CSP FILTERS
        "csp":{
            "spatial_dim": 6,  # number of CS patterns to retain (in total, per band) (K)
            "optmode": "ratiotrace"  # optimization mode: 'ratiotrace' or 'traceratio'
        }}

    if usingDataset:  # When the CSP filters and LDA is trained on a dataset (4-dimensional) and Subject Independent
        if dataset == "das-2016":
            # Inconsistente subject verwijderd (nr 7 (6+1))
            trainingSet = set(range(1, 11))-{6}

        if dataset == "dataSubject":
            # TODO: aanpassen zodat alle (consistente) subjects gebruikt worden voor training
            trainingSet = set('8')

        firstTrainingSubject = True
        print("Loading data other subjects")
        for sb in trainingSet:
            [eeg, attendedEar, fs, trialLength] = loadData(dataset, sb, params["preprocessing"],
                                                              params["conditions"])

            # Random subset in trial dimension
            ind = random.choices(list(range(0, len(attendedEar))),
                                 k=round(params["preprocessing"]["subset"] * len(attendedEar)))
            attendedEar = attendedEar[ind]
            eeg = eeg[ind, :, :]

            # apply FB
            #eerst afmetingen: shape eeg (24, 7200, 48)
            #nu afmetingen eeg: shape eeg (48, 24, 7200)
            # [0] --> [1], [1] ---> [2], [2] --> [0]
            eegTemp = eeg  #(24,7200,48) ---> (7200, 24, 48) ===== np.transpose(onze eeg)
            eeg = np.zeros((eeg.shape[0], len(params["filterbankBands"][0]), eeg.shape[1], eeg.shape[2]), dtype=np.float32)
            for band in range(len(params["filterbankBands"][0])):
                b, a = scipy.signal.iirfilter(8, np.array(
                    [2 * params["filterbankBands"][0, band] / fs, 2 * params["filterbankBands"][1, band] / fs]))
                eeg[:, band, :, :] = np.transpose(scipy.signal.filtfilt(b, a, np.transpose(eegTemp), axis=0))
                # shape eeg: trials (14) x bands (1) x channels (24) x time (7200)
                mean = np.average(eeg[:, band, :, :], 2) # shape: trials (14) x channels(24)
                means = np.full((eeg.shape[3], eeg.shape[0], eeg.shape[2]), mean) # channels(24) x trials(14) x time(7200)
                means = np.transpose(means, (1,2,0))

                eeg[:, band, :, :] = eeg[:, band, :, :] - means # trials(14) x (band(1)x) channels(24) x time(7200)
            del eegTemp
            # save results
            if firstTrainingSubject:
                X = eeg
                labels = attendedEar
                firstTrainingSubject = False
            else:
                X = np.concatenate((X, eeg), axis=3)
                labels = np.concatenate((labels, attendedEar))


    #TODO: subject specific case
    else:  # When the training data is subject specific and 2-dimensional in channels x time

        # apply FB
        eegTemp = eeg
        eeg = np.zeros((eeg.shape[0], len(params["filterbankBands"][0]), eeg.shape[1]), dtype=np.float32)
        for band in range(len(params["filterbankBands"][0])):
            b, a = scipy.signal.iirfilter(8, np.array(
                [2 * params["filterbankBands"][0, band] / fs, 2 * params["filterbankBands"][1, band] / fs]))
            eeg[:, band, :] = np.transpose(scipy.signal.filtfilt(b, a, np.transpose(eegTemp, (1, 0)), axis=0),
                                              (1, 0))
            # eeg now has dimensions channels x time x trials
            mean = np.average(eeg[:, band, :], axis=1)[:, np.newaxis]
            eeg[:, band, :] = eeg[:, band, :] - mean
        del eegTemp

        X = eeg
        labels = markers

        # Segment the EEG recording into the separate trials with a specific class in markers
        X = X[:, :, :, np.newaxis]
        X = segment(X, trialSize)
        trialLength = trialSize


    """TRAIN CSP FILTERS"""
    print("---Training CSP---")

    first = True
    CSP = dict()
    for band in range(0, len(params["filterbankBands"][0])):
        Xtrain = X[:, band, :, :]

        # Covariance weighting


        # train CSP filter
        [W, score, traceratio] = trainCSP(Xtrain, labels, params["csp"]["spatial_dim"], params["csp"]["optmode"],
                                                   params["cov"]["method"])

        if first:
            CSP["W"] = W
            CSP["score"] = score
            CSP["traceratio"] = traceratio
            #X: [trials 14, bands 1, channels 24, time 7200]
            #W: [channels 24, spatial dim 6]
            Y = [0] * np.shape(X)[0]
            for trial in range(np.shape(X)[0]):
                Y[trial] = np.dot(np.transpose(CSP["W"]), np.squeeze(X[trial, band, :, :]))

            first = False
            # Shape Y: [trials 14, spatial dim 6, time 7200]

        # TODO: Work on multiple bands
        else:
            CSP["W"] = np.concatenate((CSP["W"], W), axis=2)
            CSP["score"] = np.concatenate((CSP["score"], score), axis=2)
            CSP["traceratio"] = np.concatenate((CSP["traceratio"], traceratio), axis=2)

            # Filter both training and testing data using CSP filters
            Ytemp = [0]*np.shape(X)[0]
            for trial in range(np.shape(X)[0]):
                Ytemp[trial]= np.dot(np.transpose(CSP["W"][:, :, band]), np.squeeze(X[trial, band, :, :]))
            Y = np.concatenate((Y, Ytemp))
            # Y = np.concatenate((Y, Ytemp, axis=3))
            # Shape Y: [trials, spatial dim, time, bands]
    print("CSP TRAINING DONE")

    print("trainFilters LINE 195")

    # Y [ trials 14, spatial dim 6, time 7200, bands 1]

    if len(params["filterbankBands"][0]) == 1:
        Y = np.squeeze(Y)
        Y = Y[:, :, :, np.newaxis]
        CSP["W"] = CSP["W"][ :, :, np.newaxis]

    Y = np.transpose(Y, (0, 3, 1, 2))
    #Y [ trials 14, bands, 1, spatial dim 6, time 7200]


    """CALCULATE THE COEFFICIENTS""" #LDA
    YtrainWindow = segment(Y, windowLength * fs)
    print(YtrainWindow.shape)
    labelstrainWindow = np.repeat(labels, np.floor(trialLength / (windowLength * fs)))
    feat = np.log( np.var( YtrainWindow, axis=2 ) )
    print(feat.shape)
    feat = np.transpose(np.reshape(feat, (feat.shape[0] * feat.shape[1], feat.shape[2])))

    trainindices1 = np.where(labelstrainWindow == 1)
    trainindices2 = np.where(labelstrainWindow == 2)
    mu = np.transpose(np.array([np.average(feat[trainindices1[0], :], axis=0),
              np.average(feat[trainindices2[0], :], axis=0)]))
    print(mu.shape)

    S = np.cov(np.transpose(feat))
    print(S.shape)
    coef = np.linalg.solve(S, (np.subtract(mu[:, 1], mu[:, 0])))
    b = -np.matmul(np.transpose(coef[:, np.newaxis]), np.sum(mu, axis=1)[:, np.newaxis])*1/2
    b = np.squeeze(np.squeeze(b))


    return CSP, coef, b, feat


