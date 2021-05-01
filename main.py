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


def setup_emulator():
    # TODO: split eeg_data in left and right -> location (in file eeg_emulation)
    # TODO: this emulator code is not used yet.
    # !! Verify used OS in eeg_emulation??? Start the emulator.
    eeg_emulator = multiprocessing.Process(target=emulate)
    eeg_emulator.daemon = True
    time.sleep(5)
    eeg_emulator.start()
    # TODO: decent documentation; all info can be found in
    #  https://www.downloads.plux.info/OpenSignals/OpenSignals%20LSL%20Manual.pdf


def setup_streams():
    # SET-UP LSL Streams + resolve an EEG stream on the lab network
    print("looking for an EEG stream... ")
    streams = resolve_stream('type', 'EEG')
    print("[STREAM FOUND]")
    # create a new inlet to read from the stream
    EEG_inlet = StreamInlet(streams[0])
    return EEG_inlet


def setup_audio():
    # Volume parameters
    volumeThreshold = 10  # in percentage
    volLeft = 50  # Starting volume in percentage
    volRight = 50  # Starting volume in percentage

    # TODO: these are the ALSA related sound settings, to be replaced with
    #  by your own audio interface building block. Note that the volume
    #  controls are also ALSA specific, and need to be changed
    """ SET-UP Headphones """
    device_name = 'sysdefault'
    control_name = 'Headphone'
    cardindex = 0 #"PCH"

    wav_fn = os.path.join(os.path.expanduser('~/Desktop'), 'Pilot_1.wav')
    # Audio Control
    lr_bal = LRBalancer()
    lr_bal.set_control(control_name, device_name, cardindex)

    lr_bal.set_volume_left(volLeft)
    lr_bal.set_volume_right(volRight)

    audio = {"device_name": device_name, "cardindex": cardindex, "volLeft": volLeft, "volRight": volRight,
             "volumeThreshold": volumeThreshold, "wav_fn": wav_fn, "lr_bal": lr_bal}
    return audio


def no_training(preset):
    CSP, coefficients, b, f_in_classes, decisionWindow, samplingFrequency, filterbankband = loadData(preset,
                                                                                                     noTraining=True)
    print("Information:")
    print("DecisionWindow ", decisionWindow, " seconds")
    print("SamplingFrequency ", samplingFrequency, " Hz")
    print("FilterBankband ", filterbankband)

    data = {"CSP": CSP, "coefficients": coefficients, "b": b, "f_in_classes": f_in_classes,
            "decisionWindow": decisionWindow, "samplingFrequency": samplingFrequency, "filterbankband": filterbankband}

    return data


def data_training(trainingDataset, filtering):
    """PARAMETERS FILTERING"""
    samplingFrequency = filtering["samplingFrequency"]
    decisionWindow = filtering["decisionWindow"]
    filterbankband = filtering["filterbankband"]

    [eeg, attendedEarTraining, eegSamplingFrequency] = loadData(trainingDataset, noTraining=False)
    # eeg = eeg[:,:22,:]

    # RESAMPLING
    if eegSamplingFrequency != samplingFrequency:
        eeg = resampy.resample(eeg, eegSamplingFrequency, samplingFrequency, axis=2)

    print("-***- ", trainingDataset, " -***-")
    print("TIMEFRAME: ", decisionWindow, " SECONDS")

    if trainingDataset[:11] == "dataSubject":
        # TRAINING WITH LAST 36 MINUTES
        attendedEarTraining = attendedEarTraining[12:]
        eeg = eeg[12:, :22, :]
        # removing spikes from data
        remove_index = np.arange(samplingFrequency)
        eeg = np.delete(eeg, remove_index, axis=2)

    CSP, coefficients, b, f_in_classes = trainFilters(eeg, attendedEarTraining, fs=samplingFrequency,
                                                      filterbankBands=filterbankband, timefr=decisionWindow)

    data = {"CSP": CSP, "coefficients": coefficients, "b": b, "f_in_classes": f_in_classes,
            "decisionWindow": decisionWindow, "samplingFrequency": samplingFrequency, "filterbankband": filterbankband}
    return data


def realtime_training(audio, signal, trainingLength, filtering, save_data):
    """PARAMETERS AUDIO"""
    device_name = audio["device_name"]
    cardindex = audio["cardindex"]
    wav_fn = audio["wav_fn"]

    """PARAMETERS INCOMING SIGNAL"""
    EEG_inlet = signal["EEG_inlet"]
    datatype = signal["datatype"]
    channels = signal["channels"]
    eegSamplingFrequency = signal["eegSamplingFrequency"]

    """PARAMETERS FILTERING"""
    samplingFrequency = filtering["samplingFrequency"]
    decisionWindow = filtering["decisionWindow"]
    filterbankband = filtering["filterbankband"]
    timeframeTraining = 60 * eegSamplingFrequency  # in samples of each trial with a specific class #seconds*samplingfreq

    """PARAMETERS SAVING DATA"""
    saveTrainingData = save_data["saveTrainingData"]
    locationSavingTrainingData = save_data["locationSavingTrainingData"]

    print("Concentrate on the LEFT speaker", flush=True)
    input("Press enter to continue:")

    """Starting audio"""
    ap = AudioPlayer()
    ap.set_device(device_name, cardindex)
    ap.init_play(wav_fn)
    ap.play()
    attendedEarTraining = []
    for p in range(trainingLength):
        if p != 0 and p % 3 == 0:
            ap.pause(True)
            print("Small break")
            input("Press enter to continue:")
            ap.pause(False)
        tempeeg1, notused = receive_eeg(EEG_inlet, timeframeTraining, datatype=datatype, channels=channels)
        if p == 0:
            eeg1 = tempeeg1
            eeg1 = eeg1[np.newaxis, :]
            attendedEarTraining.append(1)
        else:
            tempeeg1 = tempeeg1[np.newaxis, :]
            eeg1 = np.concatenate([eeg1, tempeeg1])
            attendedEarTraining.append(1)
    ap.stop()

    print("Concentrate on the RIGHT speaker now", flush=True)
    input("Press enter to continue:")
    ap = AudioPlayer()
    ap.set_device(device_name, cardindex)
    ap.init_play(wav_fn)
    ap.play()
    for p in range(trainingLength):
        if p != 0 and p % 3 == 0:
            ap.pause(True)
            print("Small break")
            input("Press enter to continue:")
            ap.pause(False)
        tempeeg2, notused = receive_eeg(EEG_inlet, timeframeTraining, datatype=datatype, channels=channels)
        if p == 0:
            eeg2 = tempeeg2
            eeg2 = eeg2[np.newaxis, :]
            attendedEarTraining.append(2)
        else:
            tempeeg2 = tempeeg2[np.newaxis, :]
            eeg2 = np.concatenate([eeg2, tempeeg2])
            attendedEarTraining.append(2)
    ap.stop()

    # RESAMPLING
    if eegSamplingFrequency != samplingFrequency:
        eeg1 = resampy.resample(eeg1, eegSamplingFrequency, samplingFrequency, axis=2)
        eeg2 = resampy.resample(eeg2, eegSamplingFrequency, samplingFrequency, axis=2)

    if saveTrainingData:
        now = datetime.now()
        foldername = "Realtime TrainingData " + now.strftime("%d %b %y %H-%M")
        path_realtimedata = os.path.join(locationSavingTrainingData, foldername)
        if not os.path.exists(path_realtimedata):
            os.makedirs(path_realtimedata)

        location_fulleeg = path_realtimedata + "/eeg"
        location_attendedEar = path_realtimedata + "/attendedEar"
        location_fs = path_realtimedata + "/fs"
        np.save(location_attendedEar, attendedEarTraining)
        np.save(location_fs, samplingFrequency)
        fulleeg = np.concatenate([eeg1, eeg2])
        np.save(location_fulleeg, fulleeg)

    # Train FBCSP and LDA
    CSP, coefficients, b, f_in_classes = trainFilters(usingData=False, eeg1=eeg1, eeg2=eeg2, fs=samplingFrequency,
                                                      filterbankBands=filterbankband, timefr=decisionWindow)

    data = {"CSP": CSP, "coefficients": coefficients, "b": b, "f_in_classes": f_in_classes,
            "decisionWindow": decisionWindow, "samplingFrequency": samplingFrequency, "filterbankband": filterbankband}
    return data


def plot_features(f_in_classes):
    # "TrainingFeatures PLOT:"
    for i in range(np.shape(f_in_classes[0])[0]):
        red_scat = plt.scatter(f_in_classes[0][i][0], f_in_classes[0][i][5], color='red',
                               label='Training Class 1')
    for i in range(np.shape(f_in_classes[1])[0]):
        green_scat = plt.scatter(f_in_classes[1][i][0], f_in_classes[1][i][5], color='green',
                                 label='Training Class 2')
    # plt.legend(("Class 1", "Class 2"))
    plt.title("Feature vectors of 1st and 6th dimension plotted in 2D")
    plt.legend(handles=[red_scat, green_scat])
    # name = os.getcwd() + "/TrainingFeaturePlot"
    # plt.savefig(name)
    plt.show()
    # plt.close()


def save_CSP(locationSavingTrainingData, data):
    """PARAMETERS CSP"""
    CSP = data["CSP"]
    coefficients = data["coefficients"]
    b = data["b"]
    f_in_classes = data["f_in_classes"]
    decisionWindow = data["decisionWindow"]
    samplingFrequency = data["samplingFrequency"]
    filterbankband = data["filterbankband"]

    now = datetime.now()
    foldername = "Processed TrainingData " + now.strftime("%d %b %y %H-%M")
    path_realtimedata = os.path.join(locationSavingTrainingData, foldername)
    if not os.path.exists(path_realtimedata):
        os.makedirs(path_realtimedata)
    location_CSP = path_realtimedata + "/CSP"
    location_coefficient = path_realtimedata + "/coefficient"
    location_b = path_realtimedata + "/bias"
    location_TrainingFeatures = path_realtimedata + "/TrainingFeatures"
    location_decisionWindow = path_realtimedata + "/DecisionWindow"
    location_samplingFrequency = path_realtimedata + "/fs"
    location_filterbankband = path_realtimedata + "/FilterBankBand"
    np.save(location_CSP, CSP)
    np.save(location_coefficient, coefficients)
    np.save(location_b, b)
    np.save(location_TrainingFeatures, f_in_classes)
    np.save(location_decisionWindow, decisionWindow)
    np.save(location_samplingFrequency, samplingFrequency)
    np.save(location_filterbankband, filterbankband)


def testing(audio, signal, testingLength, filtering, classifying, save_data):
    # """PARAMETERS AUDIO"""
    # device_name = audio["device_name"]
    # cardindex = audio["cardindex"]
    # wav_fn = audio["wav_fn"]
    # lr_bal = audio["lr_bal"]
    # volumeThreshold = audio["volumeThreshold"]
    # volLeft = audio["volLeft"]
    # volRight = audio["volLeft"]

    """PARAMETERS INCOMING SIGNAL"""
    EEG_inlet = signal["EEG_inlet"]
    datatype = signal["datatype"]
    channels = signal["channels"]
    eegSamplingFrequency = signal["eegSamplingFrequency"]

    """PARAMETERS FILTERING"""
    decisionWindow = filtering["decisionWindow"]
    samplingFrequency = filtering["samplingFrequency"]
    filterbankband = filtering["filterbankband"]

    """PARAMETERS CLASSIFYING"""
    CSP = classifying["CSP"]
    coefficients = classifying["coefficients"]
    b = classifying["b"]
    f_in_classes = classifying["f_in_classes"]

    """PARAMETERS SAVING DATA"""
    saveTestingData = save_data["saveTestingData"]
    locationSavingTestingData = save_data["locationSavingTestingData"]  # only needed when saveTestingData is True

    """ System Loop """
    print('---Starting the realtime AAD testing---')
    input("Press enter to continue:")

    # TODO: dedicated plot function.
    leftOrRight_data = list()
    eeg_plot = list()
    featplot = []
    count = 0
    left = True
    false = 0
    plt.figure("Realtime EEG")
    labels = []
    previousLeftOrRight = 0
    first = True
    for nummers in range(1, 25):
        labels.append('Channel ' + str(nummers))
    attendedEarTesting = []
    # ap = AudioPlayer()
    # ap.set_device(device_name, cardindex)
    # ap.init_play(wav_fn)
    # ap.play()
    # time.sleep(2)
    while True:
        if count % 120 == 0:
            if left:
                print("Listen to the left")
                # ap.pause(True)
                # input("Press enter to continue")
                # ap.pause(False)
                attendedEarTesting.append(1)
                attendedEarTesting.append(1)
                left = False
            else:
                print("Listen to the right")
                # ap.pause(True)
                # input("Press enter to continue")
                # ap.pause(False)
                attendedEarTesting.append(2)
                attendedEarTesting.append(2)
                left = True
        # Receive EEG from LSL
        timeframe_classifying = decisionWindow * samplingFrequency
        timeframe_plot = samplingFrequency  # seconds
        for second in range(round(timeframe_classifying / samplingFrequency)):
            eeg, unused = receive_eeg(EEG_inlet, timeframe_plot, datatype=datatype, channels=channels)
            # RESAMPLING
            if eegSamplingFrequency != samplingFrequency:
                eeg = resampy.resample(eeg, eegSamplingFrequency, samplingFrequency, axis=1)
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
            eeg_to_plot = eeg[0]  # first frequencyband
            if count%60==0:
                if count == 60:
                    eeg_data = eeg_per_minute
                elif count > 60:
                    eeg_data = np.concatenate((eeg_data, eeg_per_minute), axis=0)
                eeg_per_minute = eeg
            else:
                eeg_per_minute = np.concatenate((eeg_per_minute, eeg), axis=2)

            if first:
                eeg_plot = eeg_to_plot
                classify_eeg = filtered_eeg
                first = False
            else:
                eeg_plot = np.concatenate((eeg_plot, eeg_to_plot), axis=1)
                classify_eeg = np.concatenate((classify_eeg, filtered_eeg), axis=2)

            for channel in range(len(eeg_plot)):
                eeg_plot[channel, -timeframe_plot:] = np.add(eeg_plot[channel, -timeframe_plot:],
                                                             np.full((timeframe_plot,), 20 * (len(eeg_plot) - channel)))
            eeg_plot = np.transpose(eeg_plot)
            # realtime EEG-plot:
            if len(eeg_plot) <= 5 * timeframe_plot:
                timesamples = list(np.linspace(0, count + 1, (count + 1) * timeframe_plot))
                plt.plot(timesamples, eeg_plot)
            else:
                timesamples = list(np.linspace(count - 5, count, 5 * timeframe_plot))
                plt.plot(timesamples, eeg_plot[-(5 * timeframe_plot):])
            plt.ylabel("EEG amplitude (mV)")
            plt.xlabel("time (seconds)")
            plt.title("Realtime EEG emulation")
            plt.axis([None, None, 0, 500])
            plt.legend(labels, bbox_to_anchor=(1.0, 0.5), loc="center left")
            plt.draw()
            name = os.getcwd() + "/RealtimeEegPlot"
            plt.savefig(name)
            plt.pause(1 / 120)
            plt.clf()
            eeg_plot = np.transpose(eeg_plot)
            count += 1

        # Classify eeg chunk into left or right attended speaker using CSP filters
        "---Classifying---"
        classify_eeg = np.transpose((np.transpose(classify_eeg)[-timeframe_classifying:]))
        leftOrRight, feat = classifier(classify_eeg, CSP, coefficients, b, filterbankBands=filterbankband)
        featplot.append(feat)

        # Calculating how many mistakes were made
        if leftOrRight == -1.:
            leftOrRight_data.append(1)
            print("LEFT")
            if attendedEarTesting[math.floor((count - 1) / 60)] == 2:
                false += 1
                # print("wrong ", count)
        elif leftOrRight == 1.:
            leftOrRight_data.append(2)
            print("RIGHT")
            if attendedEarTesting[math.floor((count - 1) / 60)] == 1:
                false += 1
                # print("wrong ", count)
        if count % 60 == 0:
            # print("Until minute " + str(int(count/60)) + ": " + str(false))
            plt.figure("feature")
            for i in range(np.shape(f_in_classes[0])[0]):
                green_scat = plt.scatter(f_in_classes[0][i][0], f_in_classes[0][i][5], color='darkseagreen',
                                         label='Training Class 1')
            for i in range(np.shape(f_in_classes[1])[0]):
                orange_scat = plt.scatter(f_in_classes[1][i][0], f_in_classes[1][i][5], color='orange',
                                          label='Training Class 2')
            # plt.legend(("Class 1", "Class 2"))
            plt.title("Feature vectors of 1st and 6th dimension plotted in 2D")
            f = featplot[-round(60 / decisionWindow):]
            for i in range(int(np.shape(f)[0])):
                red_scat = plt.scatter(f[i][0], f[i][5], color='red', label='Test')
            plt.legend(handles=[green_scat, orange_scat, red_scat])
            # plt.show()
            # name = "/Users/neleeeckman/Desktop/testing subjects features/"
            # name += trainingDataset[:-4] + "/TIMEFR" + str(decisionWindow) + "_MIN" + str(int(count/60))
            # name = os.getcwd() + "/FeaturePlot" + " Minute " + str(int(count/60))
            name = os.getcwd() + "/FeaturePlot"
            plt.savefig(name)
            plt.close()

            plt.figure("Realtime EEG")

        # Faded gain control towards left or right, stops when one channel falls below the volume threshold
        # Validation: previous decision is the same as this one
#        print(lr_bal.get_volume())
#        if previousLeftOrRight == leftOrRight:
#            # print("---Controlling volume---")
#            if leftOrRight == -1.:
#                if volLeft <= 75:
#                    volLeft += 25
#                elif volLeft > 75:
#                    volLeft = 100
#                if volRight >= volumeThreshold + 15:
#                    volRight -= 15
#                elif volRight < volumeThreshold + 15:
#                    volRight = volumeThreshold
#                lr_bal.set_volume_left(volLeft)
#                lr_bal.set_volume_right(volRight)

#            elif leftOrRight == 1.:
#                if volRight <= 75:
#                    volRight += 25
#                elif volRight > 75:
#                    volRight = 100
#                if volLeft >= volumeThreshold + 15:
#                    volLeft -= 15
#                elif volLeft < volumeThreshold + 15:
#                    volLeft = volumeThreshold + 15
#                lr_bal.set_volume_left(volLeft)
#                lr_bal.set_volume_right(volRight)
#        previousLeftOrRight = leftOrRight
        if count == testingLength * 60:
            # ap.stop()
            break

    print(100 - false * decisionWindow * 100 / (60 * testingLength), "%")
    # TODO: save testing data (eeg, leftOrRight, fs, ...)
    if saveTestingData:
        now = datetime.now()
        foldername = "TestingData " + now.strftime("%d %b %y %H-%M")
        path_realtimedata = os.path.join(locationSavingTestingData, foldername)
        if not os.path.exists(path_realtimedata):
            os.makedirs(path_realtimedata)

        location_fulleeg = path_realtimedata + "/eeg"
        location_attendedEar = path_realtimedata + "/attendedEar"
        location_fs = path_realtimedata + "/fs"
        np.save(location_attendedEar, leftOrRight_data)
        np.save(location_fs, eegSamplingFrequency)
        np.save(location_fulleeg, eeg_data)


path = os.getcwd()
path_trainingdata = os.path.join(os.path.join(path, "RealtimeTrainingData"), "Realtime TrainingData 04_30_21 15_46_51")
path_preset = os.path.join(os.path.join(path, "RealtimeTrainingData"), "Processed TrainingData 30_04 no audio")
path_subject8 = "dataSubject8.mat"

PARAMETERS = {"NoTraining": True, "preset": path_preset, "trainingDataset": path_trainingdata,
              "RealtimeTraining": False, "SamplingFrequency": 250, "DownSampledFrequency": 250, "Channels": 24,
              "decisionWindow": 5, "filterBankband": np.array([[12], [30]]),
              "saveTrainingData": False, "locationSavingTrainingData": os.getcwd()+"/RealtimeTrainingData",
              "saveTestingData": False, "locationSavingTestingData": os.getcwd()+"/RealtimeTestingData"}


def main(parameters):
    trainingLength = 6  # minutes
    testingLength = 4  # minutes

    # TODO necessary IN GUI
    """No Training"""
    noTraining = parameters["NoTraining"]
    preset = parameters["preset"]
    # channels = parameters["Channels"]

    """TRAINING"""
    filtering = {"filterbankband": parameters["filterBankband"],
                 "samplingFrequency": parameters["DownSampledFrequency"],  # we downsample to this fs
                 "decisionWindow": parameters["decisionWindow"]}
    realtimeTraining = parameters["RealtimeTraining"]
    save_training_data = {"saveTrainingData": parameters["saveTrainingData"],
                          "locationSavingTrainingData": parameters["locationSavingTrainingData"]}

    """Realtime training"""
    signal = {"datatype": np.float32, "channels": parameters["Channels"],
              "eegSamplingFrequency": parameters["SamplingFrequency"]}
    """Using existing eeg-file"""
    trainingDataset = parameters["trainingDataset"]  # IN GUI: als default "None"

    """Testing"""
    save_testing_data = {"saveTestingData": parameters["saveTestingData"],
                          "locationSavingTestingData": parameters["locationSavingTestingData"]}
    # channels = parameters["Channels"]


    #  Parameters that don't change --> NOT IN GUI
    retrain = True

    # # ONLY FOR CROSSVALIDATION! TODO:delete this & change emulator import:
    # [eeg, attendedEar, samplingFrequency] = loadData(trainingDataset, noTraining=False)
    # training_data, testing_data, training_attended_ear, unused = train_test_split(eeg, attendedEar, test_size=0.25)

    setup_emulator()

    signal["EEG_inlet"] = setup_streams()
    # audio = setup_audio()

    # TODO: unimplemented parameters.
    # stimulusReconstruction = False  # Use of stimulus reconstruction
    # data_subject = loadmat('dataSubject8.mat')
    # trainingDataset = np.squeeze(np.array(data_subject.get('eegTrials')))
    # Where to store eeg data in case of subject specific filtertraining:

    """"TRAINING:"""
    if noTraining:
        data = no_training(preset)
        retrain = False

    while retrain:
        # Start CSP filter and LDA training for later classification.
        print("--- Training filters and LDA... ---")
        if not realtimeTraining:  # Subject independent / dependent (own file)"
            data = data_training(trainingDataset, filtering)

            # training_data, unused, training_attended_ear, unused = train_test_split(eeg, attendedEar,test_size=0.25)
            # CSP, coefficients, b, f_in_classes = trainFilters(training_data, training_attended_ear, fs=samplingFrequency,
            # filterbankBands=filterbankband, timefr=decisionWindow)

        else:
            """Realtime training"""
            data = realtime_training(audio, signal, trainingLength, filtering, save_training_data)

        plot_features(data["f_in_classes"])

        response = input("Continue to realtime testing?"+"\r\n"+" [y/n]:")
        if response == "y" or response == "Y":
            retrain = False
        # RETRAINING:
        elif response == 'n' or response == "N":
            # REALTIME TRAINING
            if realtimeTraining:
                response = input("Do you want to continue using realtime training? [y/n]:")
                if response == 'n' or response == "N":
                    realtimeTraining = False
                    trainingDataset = input("Enter new path for training dataset")
            # NOT REALTIME TRAINING
            else:
                trainingDataset = input("Enter new path for training dataset")

    # Saving CSP, coefficents, b and features in classes:
    if save_training_data["saveTrainingData"]:
        save_CSP(save_training_data["locationSavingTrainingData"], data)

    audio = None
    testing(audio, signal, testingLength, filtering, data, save_testing_data)


if __name__ == '__main__':
    main(PARAMETERS)
