import numpy as np
import time
import math
from audio import LRBalancer
from classifier import classifier
from receive_eeg import receive_eeg
from trainFilters import trainFilters

# Parameters
samplingFrequency = 64 # Hz
timeframe = 60  # in samples (timeframe / samplingFrequency = time in seconds)
overlap = 30  # in samples
datatype = np.float32
volumeThreshold = 50  # in percent
trainingDataset = "das-2016"
updateCSP = False  # Using subject specific CSP filters
updatecov = False  # Using subject specific covariance matrix
updatebias = False  # Using subject specific bias
timeframeTraining = 36000  # in samples
trainingTrials = 10  # parts of the EEG recording. Each trial has a specific speaker class.All classes should be balanced

# SET-UP Initialize variables
leftOrRight = None

# SET-UP Headphones
device_name = 'sysdefault'
control_name = 'Headphone'
cardindex = 1

lr_bal = LRBalancer()
lr_bal.set_control(control_name, device_name, cardindex)

lr_bal.set_volume_left(100)
lr_bal.set_volume_right(100)


""" TRAINING OF THE FBCSP AND THE LDA SUBJECT INDEPENDENT OR SUBJECT SPECIFIC """
print("--- Training filters and LDA... ---")

if False in [updateCSP,updatecov,updatebias]:
    # Train the CSP filters on dataset of other subjects (subject independent)
    CSP, coef, b = trainFilters(dataset=trainingDataset)

else:
    # Update the FBCSP and LDA on eeg of the subject (subject specific)

    # Receive the eeg used for training
    eeg, markers = receive_eeg(timeframeTraining, datatype, trainingTrials)
    trialSize = math.floor(timeframeTraining / trainingTrials)
    CSPSS, coefSS, bSS = trainFilters(eeg=eeg, markers=markers, trialSize=trialSize, fs=samplingFrequency)

    """ CSP training """
    if updateCSP:
        CSP = CSPSS

    """ LDA training """
    if updatecov:
        coef = coefSS
    if updatebias:
        b = bSS


print('---Starting the system---')
while True:
    # Receive EEG from LSL
    print("---Receiving EEG---")
    eeg = receive_eeg(timeframe, datatype, overlap)

    # Classify eeg chunk into left or right attended speaker using CSP filters
    previousLeftOrRight = leftOrRight
    leftOrRight = classifier(eeg, CSP, coef, b)
    print(leftOrRight)

    # Classify eeg chunk into left or right attended speaker using stimulus reconstruction

    # Faded gain control towards left or right, stops when one channel falls below the volume threshold
    # Validation: previous decision is the same as this one

    if all(np.array(lr_bal.get_volume()) > volumeThreshold) and previousLeftOrRight == leftOrRight:
        if leftOrRight == "left":
            for i in range(3):
                lr_bal.fade_right(LRBalancer.OUT, 5)
                lr_bal.fade_left(LRBalancer.IN, 5)
                time.sleep(5)
        elif leftOrRight == "right":
            for i in range(3):
                lr_bal.fade_left(LRBalancer.OUT, 5)
                lr_bal.fade_right(LRBalancer.IN, 5)
                time.sleep(5)
