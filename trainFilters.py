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
            "subset": 1,  # subset percentage to be taken from training subjects
            "rereference": "None",
            "eegChanSel": []
        },

        #FILTERBANK SETUP
        # "filterbankBands": np.array([[1,2:2:26],[4:2:30]]),  #first row: lower bound, second row: upper bound
        "filterbankBands": np.array([[12],[30]]),

        #COVARIANCE ESTIMATION
        "cov": {
            "method": "lwcov",  # covariance matrix estimation method: 'classic' / 'lwcov'
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
            trainingSet = set(['8'])

        firstTrainingSubject = True
        print("Loading data other subjects")
        for sb in trainingSet:
            [eeg, attendedEar, fs, trialLength] = loadData(dataset, sb, params["preprocessing"],
                                                              params["conditions"])

            # Random subset in trial dimension without repetition of indices
            # ind = random.sample(list(range(0, len(attendedEar))),
            #                      k=round(params["preprocessing"]["subset"] * len(attendedEar)))
            # attendedEar = attendedEar[ind]
            # eeg = eeg[ind, :, :]
            #TRAINING WITH FIRST 36 MINUTES
            attendedEar = attendedEar[:36]
            eeg = eeg[:36, :, :]

            remove_index = np.arange(fs)
            eeg = np.delete(eeg,remove_index,axis=2)

            # apply FB
            #eerst afmetingen: shape eeg (24, 7200, 48)
            #nu afmetingen eeg: shape eeg (48, 24, 7200)
            # [0] --> [1], [1] ---> [2], [2] --> [0]
            eegTemp = eeg  #(24,7200,48) ---> (7200, 24, 48) ===== np.transpose(onze eeg)
            eeg = np.zeros((eeg.shape[0], len(params["filterbankBands"][0]), eeg.shape[1], eeg.shape[2]), dtype=np.float32)
            for band in range(len(params["filterbankBands"][0])):
                lower, upper = scipy.signal.iirfilter(8, np.array(
                    [2 * params["filterbankBands"][0, band] / fs, 2 * params["filterbankBands"][1, band] / fs]))
                eeg[:, band, :, :] = np.transpose(scipy.signal.filtfilt(lower, upper, np.transpose(eegTemp), axis=0))
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
            lower, upper = scipy.signal.iirfilter(8, np.array(
                [2 * params["filterbankBands"][0, band] / fs, 2 * params["filterbankBands"][1, band] / fs]))
            eeg[:, band, :] = np.transpose(scipy.signal.filtfilt(lower, upper, np.transpose(eegTemp, (1, 0)), axis=0),
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
    firstBand = True
    firstEEG = True
    CSP = dict()
    for band in range(0, len(params["filterbankBands"][0])):
        Xtrain = X[:, band, :, :]

        #PLOT EEG DATA:
        # print('shape Xtrain', np.shape(Xtrain))
        # plt.figure('Filtered EEG trainingdata plot')
        # # plt.plot(np.transpose(Xplot)[100:])
        # channel = 0
        # #EEG_data_plot = np.transpose(np.transpose(Xtrain[0])[100:])
        # for minute in Xtrain:
        #     newdata = np.transpose(minute)  # rows = 7200 , columns = 24 channels
        #     if firstEEG:
        #         EEG_data_plot = newdata
        #         firstEEG = False
        #     else:
        #         EEG_data_plot = np.concatenate((EEG_data_plot, newdata), axis=0)
        # EEG_data_plot = np.transpose(EEG_data_plot)
        # print("shape EEG_data_plot", np.shape(EEG_data_plot) )
        # #EEG_data_plot = Xtrain[0]
        # while channel < 24:
        #     EEG_data_plot[channel] = np.add(EEG_data_plot[channel], np.full((np.shape(EEG_data_plot)[1],), channel * (-1000)))
        #     channel += 1
        # xaxis = np.linspace(0,np.shape(Xtrain)[0], np.shape(EEG_data_plot)[1])
        # plt.plot(xaxis, np.transpose(EEG_data_plot), label='Filtered signal')
        # plt.xlabel('Time (minutes)')
        # # plt.hlines([-a, a], 0, T, linestyles='--')
        # plt.grid(True)
        # plt.axis('tight')
        # # plt.savefig('pythonfilterOrde8')
        # plt.show()


        # Covariance weighting


        # train CSP filter
        [W, score, traceratio] = trainCSP(Xtrain, labels, params["csp"]["spatial_dim"], params["csp"]["optmode"],
                                                   params["cov"]["method"])
        if firstBand:
            CSP["W"] = W
            CSP["score"] = score
            CSP["traceratio"] = traceratio
            # X: [trials 14, bands 1, channels 24, time 7200]
            # W: [channels 24, spatial dim 6]
            feat = []
            for trial in range(np.shape(X)[0]):
                Y = np.dot(np.transpose(CSP["W"]), np.squeeze(X[trial, band, :, :]))
                feat.append(logenergy(Y))

            firstBand = False
            # Shape Y: [trials 14, spatial dim 6, time 7200]

        # TODO: Work on multiple bands
        else:
            CSP["W"] = np.concatenate((CSP["W"], W), axis=2)
            CSP["score"] = np.concatenate((CSP["score"], score), axis=2)
            CSP["traceratio"] = np.concatenate((CSP["traceratio"], traceratio), axis=2)

            # Filter both training and testing data using CSP filters
            # LDA
            feat_temp = []
            for trial in range(np.shape(X)[0]):
                Ytemp = np.dot(np.transpose(CSP["W"][:, :, band]), np.squeeze(X[trial, band, :, :]))
                feat_temp.append(logenergy(Ytemp))
            feat = np.concatenate((feat, feat_temp))
            # Shape Y: [trials, spatial dim, time, bands]
            # shape feat: [trials (#minutes) , spatial dim]


    """CALCULATE THE COEFFICIENTS"""

    f_in_classes = group_by_class(feat, attendedEar)
    mean1 = np.mean(f_in_classes[0], axis=0)
    mean2 = np.mean(f_in_classes[1], axis=0)
    # ###plot training feauture###
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
    b = -0.5 * np.matmul(coef, sum_mean)

    return CSP, coef, b


