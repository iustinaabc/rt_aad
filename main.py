#!/usr/bin/env python3

import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import scipy
import time
import math
import os
import resampy

from pylsl import StreamInlet, resolve_stream, local_clock
# from audio import LRBalancer, AudioPlayer
from trainFilters import trainFilters
from receive_eeg import receive_eeg
from eeg_emulation import emulate
# from emulator_SandN import emulate
from classifier import classifier
from scipy.io import loadmat
from loadData import loadData
from group_by_class import group_by_class
from sklearn.model_selection import train_test_split
from datetime import datetime


PARAMETERS = {"RealtimeTraining": False, "SamplingFrequency": 250, "DownSampledFrequency":120, "Channels": 24,
              "trainingDataset": "/Users/neleeeckman/Documents/NeleEeckman/KULeuven/P&O BMT/rt_aad/Realtimedata/trainingdata1",
              "decisionWindow": 6, "filterBankband": np.array([[12], [30]]),
              "updateCSP": False, "updateCov": False, "updateBias": False,
              "saveTrainingData": True, "locationSavingData": os.getcwd()+"/Realtimedata"}


def main(parameters):
    trainingLength = 1 #minutes
    testingLength = 1 #minutes

    # TODO necessary IN GUI
    realtimeTraining = parameters["RealtimeTraining"]
    decisionWindow = parameters["decisionWindow"]
    filterbankband = parameters["filterBankband"]
    samplingFrequency = parameters["DownSampledFrequency"]  # we downsample to this fs

    updateCSP = parameters["updateCSP"]  # Using subject specific CSP filters
    updateCov = parameters["updateCov"]  # Using subject specific covariance matrix
    updateBias = parameters["updateBias"]  # Using subject specific bias
    saveTrainingData = parameters["saveTrainingData"]

    # enkel vragen indien saveTrainingData = True
    locationSavingData = parameters["locationSavingData"]

    #enkel vragen indien RealtimeTraining = True
    channels = parameters["Channels"]                   #IN GUI: als default "None"
    eegSamplingFrequency = parameters["SamplingFrequency"] #IN GUI: als default "None"
    # enkel vragen indien RealtimeTraining = False
    trainingDataset = parameters["trainingDataset"]     #IN GUI: als default "None"

    #  Parameters that don't change.
    datatype = np.float32


    # #ONLY FOR CROSSVALIDATION! TODO:delete this & change emulator import:
    # [eeg, attendedEar, samplingFrequency] = loadData(trainingDataset)
    # training_data, testing_data, training_attended_ear, unused = train_test_split(eeg, attendedEar, test_size=0.25)


    # TODO: split eeg_data in left and right -> location (in file eeg_emulation)
    # TODO: this emulator code is not used yet.
    # !! Verify used OS in eeg_emulation??? Start the emulator.
    eeg_emulator = multiprocessing.Process(target=emulate)
    eeg_emulator.daemon = True
    time.sleep(5)
    eeg_emulator.start()
    # TODO: decent documentation; all info can be found in
    #  https://www.downloads.plux.info/OpenSignals/OpenSignals%20LSL%20Manual.pdf

    # SET-UP LSL Streams + resolve an EEG stream on the lab network
    print("looking for an EEG stream... ")
    streams = resolve_stream('type', 'EEG')
    print("[STREAM FOUND]")
    # create a new inlet to read from the stream
    EEG_inlet = StreamInlet(streams[0])

    # TODO: unimplemented parameters.
    # stimulusReconstruction = False  # Use of stimulus reconstruction
    # data_subject = loadmat('dataSubject8.mat')
    # trainingDataset = np.squeeze(np.array(data_subject.get('eegTrials')))
    # Where to store eeg data in case of subject specific filtertraining:

    # Volume parameters
    volumeThreshold = 50  # in percentage
    volLeft = 100  # Starting volume in percentage
    volRight = 100  # Starting volume in percentage

    # TODO: these are the ALSA related sound settings, to be replaced with
    #  by your own audio interface building block. Note that the volume
    #  controls are also ALSA specific, and need to be changed
    # """ SET-UP Headphones """
    # device_name = 'sysdefault'
    # control_name = 'Headphone+LO'
    # cardindex = 0
    #
    # wav_fn = os.path.join(os.path.expanduser('~/Desktop'), 'Pilot_1.wav')
    #
    # # Playback
    # ap = AudioPlayer()
    #
    # ap.set_device(device_name, cardindex)
    # ap.init_play(wav_fn)
    # ap.play()
    #
    # # Audio Control
    # lr_bal = LRBalancer()
    # lr_bal.set_control(control_name, device_name, cardindex)
    #
    # lr_bal.set_volume_left(volLeft)
    # lr_bal.set_volume_right(volRight)

    # Start CSP filter and LDA training for later classification.
    print("--- Training filters and LDA... ---")
    if not realtimeTraining:  #Subject independent / dependent (own file)
        [eeg, attendedEar, eegSamplingFrequency] = loadData(trainingDataset)
        eegSamplingFrequency = int(eegSamplingFrequency)

        # RESAMPLING
        if eegSamplingFrequency != samplingFrequency:
            eeg = resampy.resample(eeg, eegSamplingFrequency, samplingFrequency, axis=2)

        print("-***- ", trainingDataset, " -***-")
        print("TIMEFRAME: ", decisionWindow, " SECONDS")
        # TRAINING WITH LAST 36 MINUTES
        attendedEarTraining = attendedEar[12:]
        eeg = eeg[12:, :, :]
        # removing spikes from data
        remove_index = np.arange(samplingFrequency)
        eeg = np.delete(eeg, remove_index, axis=2)

        CSP, coefficients, b, f_in_classes = trainFilters(eeg, attendedEarTraining, fs=samplingFrequency, filterbankBands=filterbankband, timefr=decisionWindow)

        # training_data, unused, training_attended_ear, unused = train_test_split(eeg, attendedEar,test_size=0.25)
        # CSP, coefficients, b, f_in_classes = trainFilters(training_data, training_attended_ear, fs=samplingFrequency, filterbankBands=filterbankband, timefr=decisionWindow)

    else:  # Realtime training
        # TODO: replace with audio player code
        # ap = AudioPlayer()
        # ap.set_device(device_name, cardIndex)
        # ap.init_play(wav_fn)
        # ap.play()

        # data_subject = loadmat(trainingDataset)
        # attended_ear = np.squeeze(np.array(data_subject.get('attendedEar')))
        # eeg_data = np.squeeze(np.array(data_subject.get('eegTrials')))
        # eeg1, eeg2 = group_by_class(eeg_data, attended_ear, 60)

        timeframeTraining = 60 * eegSamplingFrequency  # in samples of each trial with a specific class #seconds*samplingfreq

        print("Concentrate on the left speaker now", flush=True)
        input("Press enter to continue:")
        # TODO: start audio for training right ear
        attendedEarTraining = []
        startRight = local_clock()
        for p in range(trainingLength):
            tempeeg1, notused = receive_eeg(EEG_inlet, timeframeTraining, datatype=datatype, channels=channels)
            if p == 0:
                eeg1 = tempeeg1
                eeg1 = eeg1[np.newaxis, :]
                attendedEarTraining.append(1)
            else:
                tempeeg1 = tempeeg1[np.newaxis, :]
                eeg1 = np.concatenate([eeg1, tempeeg1])
                attendedEarTraining.append(1)
        # TODO: replace this with code to stop the audio player
        # ap.stop()

        # TODO: start audio for training right ear
        print("Concentrate on the right speaker now", flush=True)
        input("Press enter to continue:")
        for p in range(trainingLength):
            tempeeg2, notused = receive_eeg(EEG_inlet, timeframeTraining, datatype=datatype, channels=channels)
            if p == 0:
                eeg2 = tempeeg2
                eeg2 = eeg2[np.newaxis, :]
                attendedEarTraining.append(2)
            else:
                tempeeg2 = tempeeg2[np.newaxis, :]
                eeg2 = np.concatenate([eeg2, tempeeg2])
                attendedEarTraining.append(2)
        # TODO: replace this with code to stop the audio player
        # ap.stop()

        #RESAMPLING
        if eegSamplingFrequency != samplingFrequency:
            eeg1 = resampy.resample(eeg1, eegSamplingFrequency, samplingFrequency, axis=2)
            eeg2 = resampy.resample(eeg2, eegSamplingFrequency, samplingFrequency, axis=2)


        if saveTrainingData:
            now = datetime.now()
            foldername = now.strftime("%m:%d:%y %H.%M.%S")
            print(foldername)
            path_realtimedata = os.path.join(locationSavingData, foldername)
            if not os.path.exists(path_realtimedata):
                os.makedirs(path_realtimedata)

            location_eeg1 = path_realtimedata + "/eeg1"
            location_eeg2 = path_realtimedata + "/eeg2"
            location_fulleeg = path_realtimedata + "/fulleeg"
            location_attendedEar = path_realtimedata + "/attendedEar"
            location_fs = path_realtimedata + "/fs"
            np.save(location_eeg1, eeg1)
            np.save(location_eeg2, eeg2)
            np.save(location_attendedEar, attendedEarTraining)
            np.save(location_fs, samplingFrequency)
            fulleeg = np.concatenate([eeg1,eeg2])
            np.save(location_fulleeg, fulleeg)

        # Train FBCSP and LDA
        CSP, coefficients, b, f_in_classes = trainFilters(usingData=False, eeg1=eeg1, eeg2=eeg2, fs=samplingFrequency,
                                          filterbankBands=filterbankband, timefr=decisionWindow)

    # TODO: dedicated plot function.
    eeg_data = []
    leftOrRight_data = list()
    eeg_plot = list()
    featplot =[]

    """ System Loop """
    print('---Starting the realtime aad testing---')
    input("Press enter to continue:")
    count = 0
    false = 0
    plt.figure("Realtime EEG")
    labels = []
    previousLeftOrRight = 0
    first = True
    for nummers in range(1, 25):
        labels.append('Channel ' + str(nummers))
    [unused, attendedEarTesting, unused] = loadData(trainingDataset)
    attendedEarTesting = attendedEarTesting[:12]
    while True:
        # Receive EEG from LSL
        timeframe_classifying = decisionWindow*samplingFrequency
        timeframe_plot = samplingFrequency  # seconds
        for second in range(round(timeframe_classifying/samplingFrequency)):
            eeg, unused = receive_eeg(EEG_inlet, timeframe_plot, datatype=datatype, channels=channels)
            # RESAMPLING
            if eegSamplingFrequency != samplingFrequency:
                eeg = resampy.resample(eeg, eegSamplingFrequency, samplingFrequency, axis=2)
            '''FILTERING'''
            eegTemp = eeg  # nu afmetingen eeg: shape eeg (24, 5*120)
            eeg = np.zeros((len(filterbankband[0]), np.shape(eeg)[0], np.shape(eeg)[1]), dtype=np.float32)
            for band in range(len(filterbankband[0])):
                lower, upper = scipy.signal.iirfilter(8, np.array([2 * filterbankband[0, band] / samplingFrequency,
                                                           2 * filterbankband[1, band] / samplingFrequency]))
                eeg[band, :, :] = np.transpose(scipy.signal.filtfilt(lower, upper, np.transpose(eegTemp), axis=0))
                # shape eeg: bands (1) x channels (24) x time (7200)
                mean = np.average(eeg[band, :, :], axis=1)  # shape: channels(24)
                means = np.full((eeg.shape[2], eeg.shape[1]), mean)  # time(600) x channels(24)
                means = np.transpose(means, (1, 0))  # channels(24) x time(600)

                eeg[band, :, :] = eeg[band, :, :] - means  # (band(1)x) channels(24) x time(600)
            del eegTemp
            # save results

            filtered_eeg = eeg
            eeg_data.append(filtered_eeg)
            eeg_to_plot = eeg[0]  # first frequencyband
            # eeg_to_plot = eeg[0,4,:] # first frequency band & channel 5

            if first:
                eeg_plot = eeg_to_plot
                classify_eeg = filtered_eeg
                first = False
            else:
                eeg_plot = np.concatenate((eeg_plot, eeg_to_plot), axis=1)
                classify_eeg = np.concatenate((classify_eeg, filtered_eeg), axis=2)

            for channel in range(len(eeg_plot)):
                eeg_plot[channel, -timeframe_plot:] = np.add(eeg_plot[channel, -timeframe_plot:],
                                                             np.full((timeframe_plot,), 20*(len(eeg_plot)-channel)))
            eeg_plot = np.transpose(eeg_plot)
            # realtime EEG-plot:
            if len(eeg_plot) <= 5*timeframe_plot:
                timesamples = list(np.linspace(0, count+1, (count+1)*timeframe_plot))
                plt.plot(timesamples, eeg_plot)
            else:
                timesamples = list(np.linspace(count-5, count, 5 * timeframe_plot))
                plt.plot(timesamples, eeg_plot[-(5*timeframe_plot):])
            plt.ylabel("EEG amplitude (mV)")
            plt.xlabel("time (seconds)")
            plt.title("Realtime EEG emulation")
            plt.axis([None, None, 0, 500])
            plt.legend(labels, bbox_to_anchor=(1.0, 0.5), loc="center left")
            plt.draw()
            name = os.getcwd() + "/RealtimeEegPlot"
            plt.savefig(name)
            plt.pause(1/120)
            plt.clf()
            eeg_plot = np.transpose(eeg_plot)
            count += 1

        # Classify eeg chunk into left or right attended speaker using CSP filters
        "---Classifying---"
        classify_eeg = np.transpose((np.transpose(classify_eeg)[-timeframe_classifying:]))
        leftOrRight, feat = classifier(classify_eeg, CSP, coefficients, b, filterbankBands=filterbankband)
        leftOrRight_data.append(leftOrRight)
        featplot.append(feat)

        # Calculating how many mistakes were made
        if leftOrRight == -1.:
            print("LEFT")
            if attendedEarTesting[math.floor((count-1)/60)] == 2:
                false += 1
#                print("wrong ", count)
        elif leftOrRight == 1.:
            print("RIGHT")
            if attendedEarTesting[math.floor((count-1)/60)] == 1:
                false += 1
#                print("wrong ", count)
        if count % 60 == 0:
#            print("Until minute " + str(int(count/60)) + ": " + str(false))
            plt.figure("feature")
            for i in range(np.shape(f_in_classes[0])[1]):
                yellow_scat = plt.scatter(f_in_classes[0][i][0], f_in_classes[0][i][5], color='yellow',
                                         label='Training Class 1')
            for i in range(np.shape(f_in_classes[1])[1]):
                orange_scat = plt.scatter(f_in_classes[1][i][0], f_in_classes[1][i][5], color='orange',
                                       label='Training Class 2')
            # plt.legend(("Class 1", "Class 2"))
            plt.title("Feature vectors of 1st and 6th dimension plotted in 2D")
            f = featplot[-round(60/decisionWindow):]
            for i in range(int(np.shape(f)[0])):
                red_scat = plt.scatter(f[i][0], f[i][5], color='red', label='Test')
            plt.legend(handles=[yellow_scat, orange_scat, red_scat])
            # plt.show()
            # name = "/Users/neleeeckman/Desktop/testing subjects features/"
            # name += trainingDataset[:-4] + "/TIMEFR" + str(decisionWindow) + "_MIN" + str(int(count/60))
            name = os.getcwd() + "/FeaturePlot"
            plt.savefig(name)
            plt.close()

            plt.figure("Realtime EEG")

        # Faded gain control towards left or right, stops when one channel falls below the volume threshold
        # Validation: previous decision is the same as this one
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
        #
        #     elif leftOrRight == 1.:
        #         if volRight != 100:
        #             lr_bal.set_volume_right(100)
        #             volRight = 100
        #         print("Left Decrease")
        #         volLeft = volLeft - 5
        #         lr_bal.set_volume_left(volLeft)
        # previousLeftOrRight = leftOrRight
        if count == testingLength*60:
            break

    print(100-false*decisionWindow*100/(60*testingLength), "%")
    #TODO: save testing data (eeg, leftOrRight, fs, ...)


if __name__ == '__main__':
    main(PARAMETERS)
