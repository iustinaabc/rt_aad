import numpy as np
import scipy.signal


def classifier(eeg, CSP, coef, b):
    """
    Classifies the recorded EEG into the classes Left or Right, using the FBCSP filters and the LDA variables

    Parameters
    ----------
    :param eeg: (2-dim numpy array) The recorded EEG to be classified. Dimensions should be: channels x time samples

    :param CSP: (dict) The trained FBCSP filters saved as "W" in the dictionary

    :param coef: (1-dim numpy array) The trained LDA coefficients

    :param b: (float) The trained LDA bias

    :return: either 1 or -1 each according to a class
             can return 0 with no significance
    """

    """Parameter set up"""

    # Filterbank Set-Up
    # filterbankBands = np.array([[1,2:2:26],[4:2:30]]), #first row: lower bound, second row: upper bound
    filterbankBands = np.array([[14], [26]])

    # Sampling frequency
    fs = 64  # Hz    

    """ Apply filterbank to incoming eeg """
    eegTemp = eeg
    eeg = np.zeros((eeg.shape[0], len(filterbankBands[0]), eeg.shape[1]))
    for band in range(len(filterbankBands[0])):
        denominator, numerator = scipy.signal.iirfilter(8, np.array(
            [2 * filterbankBands[0, band] / fs, 2 * filterbankBands[1, band] / fs]))

        eeg[:, band, :] = np.transpose(scipy.signal.filtfilt(denominator, numerator, np.transpose(eegTemp, (1, 0)),
                                                             axis=0), (1, 0))
        # eeg now has dimensions channels x time
        mean = np.average(eeg[:, band, :], 1)
        means = np.full((eeg.shape[2], eeg.shape[0]), mean)
        means = np.transpose(means, (1, 0))

        eeg[:, band, :] = eeg[:, band, :] - means
    del eegTemp
    X = eeg

    """ Calculate output signal using CSP filters """
    first = True
    for band in range(len(filterbankBands[0])):
        if first:
            Y = np.matmul(np.transpose(CSP["W"][:, :, band]), np.squeeze(X[:, band, :]))
            first = False
        else:
            Y = np.concatenate((Y, np.matmul(np.transpose(CSP["W"][:, :, band]), np.squeeze(X[:, band, :]))), axis=2)

    """ Feature vector """
    feat = np.log(np.var(Y, axis=1))
    # shape should be 6xtime
    """ Prediction """
    leftOrRight = np.sign(np.matmul(feat, coef) + b)

    return leftOrRight
