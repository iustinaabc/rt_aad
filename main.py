#!/usr/bin/env python3

# import os
import numpy as np
# import scipy
import time
import math
# from audio import LRBalancer, AudioPlayer
from classifier import classifier
from receive_eeg import receive_eeg
from trainFilters import trainFilters
from pylsl import StreamInlet, resolve_stream, local_clock
import multiprocessing
from eeg_emulation import emulate

def main():
    # Parameters
    datatype = np.float32
    samplingFrequency = 250  # Hz
    channels = 24  # Channels on the EEG cap

    timeframe = 750  # in samples (timeframe / samplingFrequency = time in seconds)
    overlap = 0  # in samples

    trainingDataset = 'dataSubject8.mat'
    updateCSP = True  # Using subject specific CSP filters
    updatecov = True  # Using subject specific covariance matrix
    updatebias = True  # Using subject specific bias
    timeframeTraining = 180*samplingFrequency  # in samples of each trial with a specific class #seconds*samplingfreq
    windowLengthTraining = 10  # timeframe for training is split into windows of windowlength * fs for lda calculation
    markers = np.array([1, 2])  # First Left, then Right; for training

    #Where to store eeg data in case of subject specific filtertraining:
    location_eeg1 = '/home/rtaad/Desktop/eeg1.npy'
    location_eeg2 = '/home/rtaad/Desktop/eeg2.npy'

    # stimulusReconstruction = False  # Use of stimulus reconstruction

    # volumeThreshold = 50  # in percentage
    # volLeft = 100  # Starting volume in percentage
    # volRight = 100  # Starting volume in percentage

    dumpTrainingData = False

    # TODO: split eegdata in left and right -> location (in file eeg_emulation)
    # !! Verify used OS in eeg_emulation
    """ SET-UP Emulator """
    eeg_emulator = multiprocessing.Process(target=emulate)
    eeg_emulator.daemon = True
    time.sleep(5)
    eeg_emulator.start()

    """ SET-UP Initialize variables """
    leftOrRight = None
    eeg = None

    """ SET-UP LSL Streams """
    # resolve an EEG stream on the lab network
    print("looking for an EEG stream... ")
    streams = resolve_stream('type', 'EEG')
    print("[STREAM FOUND]")

    # create a new inlet to read from the stream
    EEG_inlet = StreamInlet(streams[0])

    # TODO: these are the ALSA related sound settings, to be replaced with
    # by your own audio interface building block. Note that the volume
    # controls are also ALSA specific, and need to be changed
#    """ SET-UP Headphones """
#    device_name = 'sysdefault'
#    control_name = 'Headphone+LO'
#    cardindex = 0
#
#    wav_fn = os.path.join(os.path.expanduser('~/Desktop'), 'Pilot_1.wav')
#
#    # Playback
#    ap = AudioPlayer()
#
#    ap.set_device(device_name, cardindex)
#    ap.init_play(wav_fn)
#    ap.play()
#
#    # Audio Control
#    lr_bal = LRBalancer()
#    lr_bal.set_control(control_name, device_name, cardindex)
#
#    lr_bal.set_volume_left(volLeft)
#    lr_bal.set_volume_right(volRight)

    """ TRAINING OF THE FBCSP AND THE LDA SUBJECT INDEPENDENT OR SUBJECT SPECIFIC """
    print("--- Training filters and LDA... ---")
    if False in [updateCSP,updatecov,updatebias]:
        # Train the CSP filters on dataset of other subjects (subject independent)
        CSP, coef, b = trainFilters(dataset=trainingDataset)

    else:
        # Update the FBCSP and LDA on eeg of the subject (subject specific)

        """ Receive the eeg used for training """
        print("Concentrate on the left speaker first", flush=True)
        startleft = local_clock()
        eeg1, timestamps1 = receive_eeg(EEG_inlet, timeframeTraining, datatype=datatype, channels=channels, starttime=startleft+3, normframe=timeframe)
        # ap.stop() # TODO: replace this with you own code to stop the audio player
        
        if dumpTrainingData:
            # DONE replace full path by a setable parameter
            np.save(location_eeg1, eeg1)

        # TODO: replace with your audio player code
        # ap = AudioPlayer()

        # ap.set_device(device_name, cardindex)
        # ap.init_play(wav_fn)
        # ap.play()

        # # TODO remove this debug code or convert into a debug option for loading
        # # rather than recording
        # # Load in previous data of own subject
        # eeg1 = np.load('/home/rtaad/Desktop/left_eeg2.npy')
        # eeg2 = np.load('/home/rtaad/Desktop/right_eeg2.npy')

        # for i in range(math.ceil(timeframeTraining/timeframe)):
        #     temp = eeg1[:, i*timeframe:(i+1)*timeframe]
        #     mean = np.average(temp, axis=1)[:, np.newaxis]
        #     temp = temp - mean
        #     eeg1[:, i*timeframe:(i+1)*timeframe] = temp
        # for i in range(math.ceil(timeframeTraining/timeframe)):
        #     temp = eeg2[:,i*timeframe:(i+1)*timeframe]
        #     mean = np.average(temp, axis=1)[:, np.newaxis]
        #     temp = temp - mean
        #     eeg2[:,i*timeframe:(i+1)*timeframe] = temp

        print("Concentrate on the right speaker now", flush=True)
        startright = local_clock()

        eeg2, timestamps2 = receive_eeg(EEG_inlet, timeframeTraining, datatype=datatype, channels=channels, starttime=startright+3, normframe=timeframe)

        if dumpTrainingData:
            # DONE replace full path by a setable parameter
            np.save(location_eeg2, eeg2)

        # TODO: better if functions take EEG1 and EEG2, rather than concatenating here
        eeg = np.concatenate((eeg1[:,15000:30000],eeg1[:,45000:],eeg2[:,15000:30000],eeg2[:,45000:]), axis=1)

        # Size of each of the two trials
        trialSize = math.floor(timeframeTraining)
        # Train FBCSP and LDA
        CSPSS, coefSS, bSS, feat = trainFilters(usingDataset=False, eeg=eeg, markers=markers, trialSize=trialSize,
                                          fs=samplingFrequency, windowLength=windowLengthTraining)

        """ CSP training """
        if updateCSP:
            CSP = CSPSS

        """ LDA training """
        if updatecov:
            coef = coefSS
        if updatebias:
            b = bSS
            print(b)
            print(coef)

    # """Test on loaded data (temporary)"""
    # scoreleft = 0
    # scoreright = 0
    # for streamcount in range(20):
    #     eegtestleft = eeg1[:,30000+streamcount*750:30000+(streamcount+1)*750]
    #     eegtestright = eeg2[:,30000+streamcount*750:30000+(streamcount+1)*750]

    #     # Classify eeg chunk into left or right attended speaker using CSP filters
    #     print("---Classifying---")
    #     previousLeftOrRight = leftOrRight
    #     leftOrRight, feat = classifier(eegtestleft, CSP, coef, b, fs=samplingFrequency)

    #     print("left" if leftOrRight == -1. else "right")
    #     if leftOrRight == -1.:
    #         scoreleft += 1
    #     leftOrRight, feat = classifier(eegtestright, CSP, coef, b, fs=samplingFrequency)
    #     if leftOrRight == 1.:
    #         scoreright += 1
    # print('left',scoreleft/20)
    # print('right',scoreright/20)

    eeg = None
    """ System Loop """
    print('---Starting the system---')
    while True:
        # Receive EEG from LSL
        print("---Receiving EEG---")
        eeg, unused = receive_eeg(EEG_inlet, timeframe, datatype=datatype, overlap=overlap, eeg=eeg, channels=channels, normframe=timeframe)

        # Classify eeg chunk into left or right attended speaker using CSP filters
        print("---Classifying---")
        # previousLeftOrRight = leftOrRight
        leftOrRight, feat = classifier(eeg, CSP, coef, b, fs=samplingFrequency)

        print("left" if leftOrRight == -1. else "right")

        # # Faded gain control towards left or right, stops when one channel falls below the volume threshold
        # # Validation: previous decision is the same as this one
        # print(lr_bal.get_volume())
        # if all(np.array(lr_bal.get_volume()) > volumeThreshold) and previousLeftOrRight == leftOrRight:
        #     print("---Controlling volume---")
        #     if leftOrRight == -1.:
        #         if volLeft != 100:
        #             lr_bal.set_volume_left(100)
        #             volLeft = 100
        #         print("Right Decrease")
        #         volRight = volRight - 5
        #         lr_bal.set_volume_right(volRight)
    
        #     elif leftOrRight == 1.:
        #         if volRight != 100:
        #             lr_bal.set_volume_right(100)
        #             volRight = 100
        #         print("Left Decrease")
        #         volLeft = volLeft - 5
        #         lr_bal.set_volume_left(volLeft)

if __name__ == '__main__':
    main()
