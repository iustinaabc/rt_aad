#!/usr/bin/env python3

import os
import numpy as np
import time
import math
from audio import LRBalancer, AudioPlayer
from classifier import classifier
from receive_eeg import receive_eeg
from trainFilters import trainFilters
from pylsl import StreamInlet, resolve_stream, local_clock
import multiprocessing
from eeg_emulation import emulate

def main():
    # Parameters
    samplingFrequency = 64 # Hz
    channels = 24 # Channels on the EEG cap
    timeframe = 100  # in samples (timeframe / samplingFrequency = time in seconds)
    overlap = 30  # in samples
    datatype = np.float32
    volumeThreshold = 50  # in percent
    trainingDataset = "das-2016"
    updateCSP = True  # Using subject specific CSP filters
    updatecov = True  # Using subject specific covariance matrix
    updatebias = True  # Using subject specific bias
    timeframeTraining = 768  # in samples of each trial with a specific class
    windowLengthTraining = 2  # timeframe for training is split into windows of windowlength * fs for lda calculation
    # trainingTrials = 1  # parts of the EEG recording. Each trial has a specific speaker class.All classes should be balanced
    stimulusReconstruction = False  # Use of stimulus reconstruction
    markers = np.array([1,2]) # First Left, then Right for training

    # Emulator SET-UP
    # eeg_emulator = multiprocessing.Process(target=emulate)
    # eeg_emulator.daemon = True
    # time.sleep(5)
    # eeg_emulator.start()

    # SET-UP Initialize variables
    leftOrRight = None
    eeg = None

    # SET-UP LSL Streams
    # resolve an EEG stream on the lab network
    print("looking for an EEG stream... ")
    streams = resolve_stream('type', 'EEG')
    print("[STREAM FOUND]")

    # create a new inlet to read from the stream
    EEG_inlet = StreamInlet(streams[0])

    # SET-UP Headphones
    device_name = 'sysdefault'
    control_name = 'Headphone+LO'
    cardindex = 0

    wav_fn = os.path.join(os.path.expanduser('~/Desktop'), 'Pilot_1.wav')

    ap = AudioPlayer()

    ap.set_device(device_name, cardindex)
    ap.init_play(wav_fn)
    ap.play()

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
        print("Concentrate on the left speaker first", flush=True)
        eeg1, timestamps1 = receive_eeg(EEG_inlet, timeframeTraining, datatype=datatype, channels=channels)

        print("Concentrate on the right speaker now", flush=True)
        # Wasted eeg while subject directs attention
        flag = True
        while flag:
            _, stamp = receive_eeg(EEG_inlet,1)
            print(stamp, local_clock(), EEG_inlet.time_correction())
        eeg2, timestamps2 = receive_eeg(EEG_inlet, timeframeTraining, datatype=datatype, channels=channels)
        timestamps = np.concatenate((timestamps1, timestamps2))
        eeg = np.concatenate((eeg1, eeg2), axis=1)
        print(eeg.shape)

        trialSize = math.floor(timeframeTraining)
        CSPSS, coefSS, bSS = trainFilters(usingDataset=False, eeg=eeg, markers=markers, trialSize=trialSize, fs=samplingFrequency, windowLength=windowLengthTraining)
        print(coefSS)
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
        eeg, unused = receive_eeg(EEG_inlet, timeframe, datatype=datatype, overlap=overlap, eeg=eeg, channels=channels)
        eeg = np.array(eeg)
        # Classify eeg chunk into left or right attended speaker using CSP filters
        print("---Classifying---")
        print(eeg.shape)
        previousLeftOrRight = leftOrRight
        leftOrRight = classifier(eeg, CSP, coef, b)
        print("left" if leftOrRight == 1. else "right")

        # Classify eeg chunk into left or right attended speaker using stimulus reconstruction

        # Faded gain control towards left or right, stops when one channel falls below the volume threshold
        # Validation: previous decision is the same as this one
        print(lr_bal.get_volume(), previousLeftOrRight, leftOrRight)

        if all(np.array(lr_bal.get_volume()) > volumeThreshold) and previousLeftOrRight == leftOrRight:
            print("---Controlling volume---")
            if leftOrRight == -1.:
                print("Left Increase")
                for i in range(3):
                    lr_bal.fade_right(LRBalancer.OUT, 5)
                    lr_bal.fade_left(LRBalancer.IN, 5)
                    time.sleep(5)
            elif leftOrRight == 1.:
                print("Right increase")
                for i in range(3):
                    lr_bal.fade_left(LRBalancer.OUT, 5)
                    lr_bal.fade_right(LRBalancer.IN, 5)
                    time.sleep(5)


if __name__ == '__main__':
    main()
