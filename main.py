#!/usr/bin/env python3

import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import scipy
import time
import math
import os

from pylsl import StreamInlet, resolve_stream, local_clock
# from audio import LRBalancer, AudioPlayer
from trainFilters import trainFilters
from receive_eeg import receive_eeg
from eeg_emulation import emulate
from classifier import classifier
from scipy.io import loadmat
from loadData import loadData

PARAMETERS = {"datatype": np.float32, "samplingFrequency": 120, "channels": 24,
              "timeframe": 7200, "overlap": 0, "trainingDataset": "dataSubject",
              "updateCSP": False, "updateCov": False, "updateBias": False,
              "windowLengthTraining": 10, "location_eeg1": "/home/rtaad/Desktop/eeg1.npy",
              "location_eeg2": "/home/rtaad/Desktop/eeg2.npy", "dumpTrainingData": False}


def main(parameters):

    # Parameter initialisation.
    # TODO: which parameters are (un)necessary?
    print(parameters)
    datatype = parameters["datatype"]  # ???
    samplingFrequency = parameters["samplingFrequency"]  # Sampling frequency in Hertz.
    channels = parameters["channels"]  # Number of electrodes on the EEG-cap.
    timeframe = parameters["timeframe"] # in samples (timeframe 7200 / samplingFrequency 120 = time in seconds = 60s)
    overlap = parameters["overlap"]  # in samples
    trainingDataset = parameters["trainingDataset"]  # File containing training data.
    updateCSP = parameters["updateCSP"]  # Using subject specific CSP filters
    updateCov = parameters["updateCov"]  # Using subject specific covariance matrix
    updateBias = parameters["updateBias"]  # Using subject specific bias
    windowLengthTraining = parameters["windowLengthTraining"]  # timeframe for training is split into windows of windowlength * fs for lda calculation
    location_eeg1 = parameters["location_eeg1"]
    location_eeg2 = parameters["location_eeg2"]
    dumpTrainingData = parameters["dumpTrainingData"]

    # Parameters that don't change.
    markers = np.array([1, 2])  # First Left, then Right; for training
    timeframeTraining = 180*samplingFrequency  # in samples of each trial with a specific class #seconds*samplingfreq

    # TODO: unimplemented parameters.
    # stimulusReconstruction = False  # Use of stimulus reconstruction
    # volumeThreshold = 50  # in percentage
    # volLeft = 100  # Starting volume in percentage
    # volRight = 100  # Starting volume in percentage
    # data_subject = loadmat('dataSubject8.mat')
    # trainingDataset = np.squeeze(np.array(data_subject.get('eegTrials')))
    # Where to store eeg data in case of subject specific filtertraining:

    # TODO: split eeg_data in left and right -> location (in file eeg_emulation)
    # TODO: this emulator code is not used yet.
    # !! Verify used OS in eeg_emulation??? Start the emulator.
    eeg_emulator = multiprocessing.Process(target=emulate)
    eeg_emulator.daemon = True
    time.sleep(5)
    eeg_emulator.start()

    # TODO: decent documentation; all info can be found in
    #  https://www.downloads.plux.info/OpenSignals/OpenSignals%20LSL%20Manual.pdf
    leftOrRight = None
    eeg = None
    # SET-UP LSL Streams + resolve an EEG stream on the lab network
    print("looking for an EEG stream... ")
    streams = resolve_stream('type', 'EEG')
    print("[STREAM FOUND]")
    # create a new inlet to read from the stream
    EEG_inlet = StreamInlet(streams[0])

    # TODO: unfinished sound code was moved from here to file "Sound".
    # sound()?

    # Start CSP filter and LDA training for later classification.
    print("--- Training filters and LDA... ---")
    if False in [updateCSP, updateCov, updateBias]:
        CSP, coefficients, b = trainFilters(trainingDataset)  # Subject independent.
    else:
        print("Concentrate on the left speaker first", flush=True)
        startLeft = local_clock()
        eeg1, timestamps1 = receive_eeg(EEG_inlet, timeframeTraining, datatype=datatype, channels=channels,
                                        starttime=startLeft+3, normframe=timeframe)  # Subject dependent.
        # TODO: replace this with you own code to stop the audio player
        # ap.stop()
        if dumpTrainingData:
            np.save(location_eeg1, eeg1)

        print("Concentrate on the right speaker now", flush=True)
        startRight = local_clock()
        eeg2, timestamps2 = receive_eeg(EEG_inlet, timeframeTraining, datatype=datatype, channels=channels,
                                        starttime=startRight+3, normframe=timeframe)
        if dumpTrainingData:
            np.save(location_eeg2, eeg2)

        # TODO: replace with your audio player code
        # ap = AudioPlayer()
        # ap.set_device(device_name, cardIndex)
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

        # TODO: better if functions take EEG1 and EEG2, rather than concatenating here
        eeg = np.concatenate((eeg1[:,15000:30000],eeg1[:,45000:],eeg2[:,15000:30000],eeg2[:,45000:]), axis=1)
        # Size of each of the two trials
        trialSize = math.floor(timeframeTraining)
        # Train FBCSP and LDA

        CSPSS, coefSS, bSS, feat = trainFilters(usingDataset=False, eeg=eeg, markers=markers,
                                                trialSize=trialSize, fs=samplingFrequency,
                                                windowLength=windowLengthTraining)

        # Train the CSP.
        if updateCSP:
            CSP = CSPSS

        # Train the LDA.
        if updateCov:
            coefficients = coefSS
        if updateBias:
            b = bSS
            print(b)
            print(coefficients)

    # TODO: scoring system to evaluate decoders.
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

    # TODO: dedicated plot function.
    eeg_data = []
    leftOrRight_data = list()
    eeg_plot = list()
    """ System Loop """
    print('---Starting the system---')
    count = 0
    plt.figure("Realtime EEG")
    labels = []
    first = True
    for nummers in range(1, 25):
        labels.append('Channel ' + str(nummers))
    while True:
        # Receive EEG from LSL
        #print("---Receiving EEG---")
        timeframe_classifying = 10*samplingFrequency
        timeframe_plot = samplingFrequency # seconds
        ##timeframe = 7200 => eeg_data [minutes, channels(24), trials(7200)]
        #timeframe = 120 => eeg_data [seconds, channels(24), trials(120)]
        for second in range(round(timeframe_classifying/samplingFrequency)):
            eeg, unused = receive_eeg(EEG_inlet, timeframe_plot, datatype=datatype, overlap=overlap, channels=channels)

            '''FILTERING'''
            params = {# FILTERBANK SETUP
                # "filterbankBands": np.array([[1,2:2:26],[4:2:30]]),  #first row: lower bound, second row: upper bound
                "filterbankBands": np.array([[12], [30]])}
            eegTemp = eeg  # nu afmetingen eeg: shape eeg (24, 5*120)
            eeg = np.zeros((len(params["filterbankBands"][0]), np.shape(eeg)[0], np.shape(eeg)[1]), dtype=np.float32)

            for band in range(len(params["filterbankBands"][0])):
                lower, upper = scipy.signal.iirfilter(8, np.array([2 * params["filterbankBands"][0, band] / samplingFrequency,
                                                           2 * params["filterbankBands"][1, band] / samplingFrequency]))
                eeg[band, :, :] = np.transpose(scipy.signal.filtfilt(lower, upper, np.transpose(eegTemp), axis=0))
                # shape eeg: bands (1) x channels (24) x time (7200)
                mean = np.average(eeg[band, :, :], axis=1)  # shape: channels(24)
                means = np.full((eeg.shape[2], eeg.shape[1]), mean)  # time(600) x channels(24)
                means = np.transpose(means, (1, 0)) # channels(24) x time(600)

                eeg[band, :, :] = eeg[band, :, :] - means  #(band(1)x) channels(24) x time(600)
            del eegTemp
            # save results

            filtered_eeg = eeg
            eeg_data.append(filtered_eeg)
            eeg_to_plot = eeg[0] # first frequencyband
            # eeg_to_plot = eeg[0,4,:] # first frequency band & channel 5

            if first:
                eeg_plot = eeg_to_plot
                classify_eeg = filtered_eeg
                first = False
            else:
                eeg_plot = np.concatenate((eeg_plot, eeg_to_plot), axis=1)
                classify_eeg = np.concatenate((classify_eeg, filtered_eeg), axis=2)

            for channel in range(len(eeg_plot)):
                eeg_plot[channel,-timeframe_plot:]= np.add(eeg_plot[channel,-timeframe_plot:], np.full((timeframe_plot,), 20*(len(eeg_plot)-channel)))
            eeg_plot = np.transpose(eeg_plot)
            # realtime EEG-plot:
            if len(eeg_plot) < 5*timeframe_plot:
                timesamples = list(np.linspace(0, count+1, (count+1)*timeframe_plot))
                plt.plot(timesamples,eeg_plot)
            else:
                timesamples = list(np.linspace(count-5, count, 5 * timeframe_plot))
                plt.plot(timesamples, eeg_plot[(-5*timeframe_plot):])
            plt.ylabel("EEG amplitude (mV)")
            plt.xlabel("time (seconds)")
            plt.title("Realtime EEG emulation")
            plt.axis([None, None, 0 ,500])
            plt.legend(labels, bbox_to_anchor=(1.0, 0.5), loc="center left")
            plt.draw()
            plt.pause(1/120)
            plt.clf()
            eeg_plot = np.transpose(eeg_plot)
            count += 1

        # Classify eeg chunk into left or right attended speaker using CSP filters
        "---Classifying---"
        classify_eeg = np.transpose((np.transpose(classify_eeg)[-timeframe_classifying:]))
        leftOrRight, feat = classifier(classify_eeg, CSP, coefficients, b, fs=samplingFrequency)
        leftOrRight_data.append(leftOrRight[0])

        print("second --- ", count)
        if leftOrRight == -1.:
            print("[LEFT]")
            #print(leftOrRight)
        elif leftOrRight == 1.:
            print("[RIGHT]")
            #print(leftOrRight)

        if count%60 == 0:
            minute = round(count/60) +1
            print("------  MINUTE ", str(minute), " -------" )

        if count == 12*60:
            break

    print(leftOrRight_data)
    # data = loadmat('dataSubject8.mat')
    # attendedEar = np.squeeze(np.array(data.get('attendedEar')))
    # print(attendedEar[:12])

    
if __name__ == '__main__':
    main(PARAMETERS)
