import tkinter as tk
from tkinter import *

import GUIFunctions
import RealTimePage
import PresetPage
import ResearcherPage
import StartPage
import TrainFilePage
import TrainStreamPage
import TrainingPage
import TrainingRoot
import UserPage

largeFont = ("Verdana", 12)
normalFont = ("Verdana", 10)


# Page for training models based on (pre)recorded data.
class TrainingPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.widgets(controller)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)
        self.rowconfigure(3, weight=1)
        self.rowconfigure(4, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self.columnconfigure(3, weight=1)
        self.columnconfigure(4, weight=1)
        self.columnconfigure(5, weight=1)

    """ Ask the user for a file and store it as a variable. """
    @staticmethod
    def askDirectory(self, var):
        direc = str(tk.filedialog.askdirectory()).replace("/", "\\")
        if direc:
            var.set(direc)

    "Write the given data to a preset file"
    def savePreset(self, data, name):
        name += ".txt"
        f = open(name, "w")
        f.write(str(data))
        f.close()

    """ Convert binary boolean values to actual boolean values. """
    def widgets(self, controller):
        def trueFalse(var):
            if var == 0:
                return False
            else:
                return True

        # Read user input values.
        def collectParameters():
            data = {}
            print("datatype: np.float32")
            data.__setitem__("datatype", "np.float32")
            samplingFrequency = samplingFrequencyEntry.get()
            print("samplingFrequency: " + samplingFrequency)
            data.__setitem__("samplingFrequency", samplingFrequency)
            channels = channelsEntry.get()
            print("channels: " + channels)
            data.__setitem__("channels", channels)
            timeframe = timeframeEntry.get()
            print("timeframe: " + timeframe)
            data.__setitem__("timeframe", timeframe)
            overlap = overlapEntry.get()
            print("overlap: " + overlap)
            data.__setitem__("overlap", overlap)
            trainingDataset = trainingDatasetVar.get()
            print("trainingDataset: " + str(trainingDataset))
            data.__setitem__("trainingDataset", trainingDataset)
            updateCSP = trueFalse(CSPCheck.get())
            print("updateCSP: " + str(updateCSP))
            data.__setitem__("updateCSP", updateCSP)
            updateCov = trueFalse(CovCheck.get())
            print("updateCov: " + str(updateCov))
            data.__setitem__("updateCov", updateCov)
            updateBias = trueFalse(BiasCheck.get())
            print("updateBias: " + str(updateBias))
            data.__setitem__("updateBias", updateBias)
            windowLengthTraining = windowLengthTrainingEntry.get()
            print("windowLengthTraining: " + windowLengthTraining)
            data.__setitem__("windowLengthTraining", windowLengthTraining)
            location_eeg1 = location_eeg1Var.get()
            print("location_eeg1: " + str(location_eeg1))
            data.__setitem__("location_eeg1", location_eeg1)
            location_eeg2 = location_eeg2Var.get()
            print("location_eeg2: " + str(location_eeg2))
            data.__setitem__("location_eeg2", location_eeg2)
            dumpTrainingData = trueFalse(dumpTrainingDataVar.get())
            print("dumpTrainingData: " + str(dumpTrainingData))
            data.__setitem__("dumpTrainingData", dumpTrainingData)
            print("\r\n")

            #mainParameters = [datatype, samplingFrequency, channels, timeframe, overlap, trainingDataset, updateCSP,
            #                  updateCov, updateBias, windowLengthTraining, location_eeg1, location_eeg2,
            #                  dumpTrainingData, None, None]
            #controller.show_frame(ResearcherPage)
            #main_process = multiprocessing.Process(target=main.main, args=(main.PARAMETERS,))
            #main_process.start()
            return data

        backgroundcolorone = "#8A9097"
        backgroundcolortwo = "#575F6B"
        textcolor = "#FFFFFF"

        presetLabel = tk.Label(self, text="Load preset:", font=largeFont,
                                 background=backgroundcolortwo, relief=RAISED, fg=textcolor)
        presetLabel.grid(row=0, column=0, sticky="nsew")
        presetVar = StringVar()
        presetButton = tk.Button(self, text='Preset', font=largeFont,
                                        command=lambda: TrainingPage.askDirectory(self, presetVar)
                                        , background=backgroundcolorone, relief=GROOVE)
        presetButton.grid(row=0, column=1, sticky="nsew")

        samplingFrequencyLabel = tk.Label(self, text="samplingFrequency:", font=largeFont,
                                          background=backgroundcolortwo, relief=RAISED, fg=textcolor)
        samplingFrequencyLabel.grid(row=0, column=2, sticky="nsew")
        samplingFrequencyEntry = tk.Entry(self, font=largeFont,
                                          background=backgroundcolorone, relief=GROOVE)
        samplingFrequencyEntry.insert(END, 120)
        samplingFrequencyEntry.grid(row=0, column=3, sticky="nsew")

        channelsLabel = tk.Label(self, text="channels:", font=largeFont,
                                 background=backgroundcolortwo, relief=RAISED, fg=textcolor)
        channelsLabel.grid(row=0, column=4, sticky="nsew")
        channelsEntry = tk.Entry(self, font=largeFont,
                                 background=backgroundcolorone, relief=GROOVE)
        channelsEntry.insert(END, 24)
        channelsEntry.grid(row=0, column=5, sticky="nsew")

        timeframeLabel = tk.Label(self, text="timeframe:", font=largeFont,
                                  background=backgroundcolortwo, relief=RAISED, fg=textcolor)
        timeframeLabel.grid(row=1, column=0, sticky="nsew")
        timeframeEntry = tk.Entry(self, font=largeFont,
                                  background=backgroundcolorone, relief=GROOVE)
        timeframeEntry.insert(END, 7200)
        timeframeEntry.grid(row=1, column=1, sticky="nsew")

        overlapLabel = tk.Label(self, text="overlap:", font=largeFont,
                                background=backgroundcolortwo, relief=RAISED, fg=textcolor)
        overlapLabel.grid(row=1, column=2, sticky="nsew")
        overlapEntry = tk.Entry(self, font=largeFont,
                                background=backgroundcolorone, relief=GROOVE)
        overlapEntry.insert(END, 0)
        overlapEntry.grid(row=1, column=3, sticky="nsew")

        trainingDatasetLabel = tk.Label(self, text="trainingDataset:", font=largeFont,
                                        background=backgroundcolortwo, relief=RAISED, fg=textcolor)
        trainingDatasetLabel.grid(row=1, column=4, sticky="nsew")
        trainingDatasetVar = StringVar()
        trainingDatasetButton = tk.Button(self, text='Choose directory', font=largeFont,
                                          command=lambda: TrainingPage.askDirectory(self, trainingDatasetVar),
                                          background=backgroundcolorone, relief=GROOVE)
        trainingDatasetButton.grid(row=1, column=5, sticky="nsew")

        updateCSPLabel = tk.Label(self, text="updateCSP?", font=largeFont,
                                  background=backgroundcolortwo, relief=RAISED, fg=textcolor)
        updateCSPLabel.grid(row=2, column=0, sticky="nsew")
        CSPCheck = tk.IntVar()
        updateCSPCheck = tk.Checkbutton(self, variable=CSPCheck, font=largeFont,
                                        background=backgroundcolorone, relief=GROOVE)
        updateCSPCheck.grid(row=2, column=1, sticky="nsew")

        updateCovLabel = tk.Label(self, text="updateCov?", font=largeFont,
                                  background=backgroundcolortwo, relief=RAISED, fg=textcolor)
        updateCovLabel.grid(row=2, column=2, sticky="nsew")
        CovCheck = tk.IntVar()
        updateCovCheck = tk.Checkbutton(self, variable=CovCheck, font=largeFont,
                                        background=backgroundcolorone, relief=GROOVE)
        updateCovCheck.grid(row=2, column=3, sticky="nsew")

        updateBiasLabel = tk.Label(self, text="updateBias?", font=largeFont,
                                   background=backgroundcolortwo, relief=RAISED, fg=textcolor)
        updateBiasLabel.grid(row=2, column=4, sticky="nsew")
        BiasCheck = tk.IntVar()
        updateBiasCheck = tk.Checkbutton(self, variable=BiasCheck, font=largeFont,
                                         background=backgroundcolorone, relief=GROOVE)
        updateBiasCheck.grid(row=2, column=5, sticky="nsew")

        windowLengthTrainingLabel = tk.Label(self, text="windowLengthTraining:", font=largeFont,
                                             background=backgroundcolortwo, relief=RAISED, fg=textcolor)
        windowLengthTrainingLabel.grid(row=3, column=0, sticky="nsew")
        windowLengthTrainingEntry = tk.Entry(self, font=largeFont,
                                             background=backgroundcolorone, relief=GROOVE)
        windowLengthTrainingEntry.insert(END, 10)
        windowLengthTrainingEntry.grid(row=3, column=1, sticky="nsew")

        location_eeg1Label = tk.Label(self, text="EEG1 location:", font=largeFont,
                                      background=backgroundcolortwo, relief=RAISED, fg=textcolor)
        location_eeg1Label.grid(row=3, column=2, sticky="nsew")
        location_eeg1Var = StringVar()
        location_eeg1Button = tk.Button(self, text='EEG1', font=largeFont,
                                        command=lambda: TrainingPage.askDirectory(self, location_eeg1Var)
                                        , background=backgroundcolorone, relief=GROOVE)
        location_eeg1Button.grid(row=3, column=3, sticky="nsew")

        location_eeg2Label = tk.Label(self, text="EEG2 location:", font=largeFont,
                                      background=backgroundcolortwo, relief=RAISED, fg=textcolor)
        location_eeg2Label.grid(row=3, column=4, sticky="nsew")
        location_eeg2Var = StringVar()
        location_eeg2Button = tk.Button(self, text='EEG2', font=largeFont,
                                        command=lambda: TrainingPage.askDirectory(self, location_eeg2Var),
                                        background=backgroundcolorone, relief=GROOVE)
        location_eeg2Button.grid(row=3, column=5, sticky="nsew")

        dumpTrainingDataLabel = tk.Label(self, text="dumpTrainingData?", font=largeFont,
                                         background=backgroundcolortwo, relief=RAISED, fg=textcolor)
        dumpTrainingDataLabel.grid(row=4, column=0, sticky="nsew")
        dumpTrainingDataVar = tk.IntVar()
        dumpTrainingDataCheck = tk.Checkbutton(self, variable=dumpTrainingDataVar, font=largeFont,
                                               background=backgroundcolorone, relief=GROOVE)
        dumpTrainingDataCheck.grid(row=4, column=1, sticky="nsew")

        saveEntry = tk.Entry(self, text="Save settings", font=largeFont,
                                 background=backgroundcolortwo, relief=RAISED, fg=textcolor)
        saveEntry.grid(row=4, column=2, sticky="nsew")
        saveButton = tk.Button(self, text="Save preset", font=largeFont,
                               command=lambda: self.savePreset(collectParameters(), saveEntry.get()),
                               background=backgroundcolorone, relief=GROOVE)
        saveButton.grid(row=4, column=3, sticky="nsew")

        startLabel = tk.Label(self, text="Start training:", font=largeFont,
                              background=backgroundcolortwo, relief=RAISED, fg=textcolor)
        startLabel.grid(row=4, column=4, sticky="nsew")
        startButton = tk.Button(self, text="START", command=lambda: [collectParameters(),
                                                                     controller.show_frame(ResearcherPage)],
                                background=backgroundcolorone, relief=GROOVE,
                                font=largeFont)
        startButton.grid(row=4, column=5, sticky="nsew")
