import numpy as np
import scipy.io as spio
from classifier import classifier
from trainFilters import trainFilters
import unittest

CSP, coef, b = trainFilters(usingDataset=True, dataset="das-2016")
mat = spio.loadmat('C:\\Users\\Xander\\Documents\\Project\\Code matlab\\code-xander\\zdata\\dataSubject3.mat')

attendedEar = mat["attendedEar"][0]
eegTrials = np.array(mat["eegTrials"])
eegTrials = np.squeeze(eegTrials)
eeg = np.zeros((eegTrials.shape[0], eegTrials[0].shape[0], eegTrials[0].shape[1]))
for i in range(eeg.shape[0]):
    eeg[i] = eegTrials[i]
eeg = np.transpose(eeg, (2, 1, 0))


def classier_test(i,eeg,CSP,coef,b,attendedEar):
    leftOrRight = classifier(eeg[:,:60,i], CSP, coef,b)
    print(leftOrRight,"samen met",attendedEar[i])

class TestClassifier(unittest.TestCase):

    def t(self):
        for i in range(5):
            lOrR = classier_test(i,eeg,CSP,coef,b,attendedEar)
