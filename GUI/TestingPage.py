import tkinter as tk
from tkinter import *
import numpy as np
import os

import main
import ResearcherPage
import StartPage

largeFont = ("Verdana", 12)
normalFont = ("Verdana", 10)


class TestingPage(tk.Frame):
    """
    Page for testing a pretrained decoder.
    """
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.widgets(controller)
        self.args = None
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

    def widgets(self, controller):

        background_color_one = "#8A9097"
        background_color_two = "#575F6B"
        text_color = "#FFFFFF"

        # Navigation buttons to other main pages.
        user_button = tk.Button(self, text="Researcher page",
                                command=lambda: controller.show_frame(ResearcherPage.ResearcherPage),
                                background=background_color_two, fg=text_color, font=largeFont)
        user_button.grid(row=0, column=0, sticky="nsew")
        home_button = tk.Button(self, text="Start page",
                                command=lambda: controller.show_frame(StartPage.StartPage),
                                background=background_color_two, fg=text_color, font=largeFont)
        home_button.grid(row=0, column=1, sticky="nsew")

        text = tk.Label(self, text="nest")
        text.grid(row=0, column=2, stick="nsew")

        self.audio_trainer = tk.Text(self, width=40, height=10, background=background_color_one, fg=text_color)
        self.audio_trainer.grid(row=1, column=0, sticky="nsew")
        test_btn = tk.Button(self, text="klik", command=lambda: self.test())
        test_btn.grid(row=1, column=1, sticky="nsew")

    def test(self):
        parameters = main.PARAMETERS

        audio = None

        signal_parameters = {"datatype": np.float32,
                             "channels": parameters["Channels"],
                             "eegSamplingFrequency": parameters["SamplingFrequency"],
                             "EEG_inlet": main.setup_streams()}

        testing_length = 4

        filter_parameters = {"filterbankband": parameters["filterBankband"],
                             "samplingFrequency": parameters["DownSampledFrequency"],
                             "decisionWindow": parameters["decisionWindow"]}

        data = self.args

        save_parameters = {"saveTrainingData": parameters["saveTrainingData"],
                     "locationSavingTrainingData": parameters["locationSavingTrainingData"]}

        main.testing(audio, signal_parameters, testing_length, filter_parameters, data, save_parameters)
