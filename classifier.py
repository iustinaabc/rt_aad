import numpy as np
import scipy.signal

def logenergy(y):
    outputEnergyVector = np.zeros(len(y))
    for i in range(len(y)):
        outputEnergyVector[i] = sum(j**2 for j in y[i])
    return np.log(outputEnergyVector)


def classifier(eeg, CSP, coef, b, filterbankBands):
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

    """ Calculating output signal and Feature vector using CSP filters """
    first = True
    for band in range(len(filterbankBands[0])):
        if first:
            # X: [bands 1, channels 24, time 600]
            # W: [band, channels 24, spatial dim 6]
            Y = np.dot(np.transpose(CSP["W"][band]), np.squeeze(eeg[band]))
            feat = logenergy(Y)

            first = False
            # Shape feat: [spatial dim 6, time 7200]
        else:
            Ytemp = np.dot(np.transpose(CSP["W"][band]), np.squeeze(eeg[band]))
            feat_temp = logenergy(Ytemp)
            feat = np.concatenate((feat, feat_temp))

    """ Prediction """
    leftOrRight = np.sign(np.matmul(coef, np.squeeze(feat)) + b)

    return leftOrRight, feat
