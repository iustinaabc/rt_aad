import tkinter as tk
from tkinter import *
import os

import main
import TrainingRoot
import TestingPage

largeFont = ("Verdana", 12)
normalFont = ("Verdana", 10)


def toTestingPhase(controller):
    path = os.getcwd()
    path_training_data = os.path.join(os.path.join(path, "RealtimeTrainingData"),
                                     "Realtime TrainingData 04:30:21 15.46.51")
    path_preset = os.path.join(os.path.join(path, "RealtimeTrainingData"),
                               "Processed TrainingData 04:30:21 15.48.06")
    path_subject8 = "dataSubject8.mat"
    test_path = r"C:\Users\gebruiker\PycharmProjects\2204\RealtimeTrainingData\Processed TrainingData with audio 30_04"

    data = main.no_training(test_path)
    controller.show_frame(TestingPage.TestingPage, data)


class PresetPage(tk.Frame):
    """
    Page that follows the main loop with "NoTraining" set to True as to not use real time training.
    """
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.widgets(controller)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

    def widgets(self, controller):

        background_color_one = "#8A9097"
        background_color_two = "#575F6B"
        text_color = "#FFFFFF"

        preset_label = tk.Label(self, text="Load preset decoder and advance to testing phase.",
                                background=background_color_one)
        preset_label.grid(row=0, column=0, sticky="nsew")
        preset_button = tk.Button(self, text="OK", command=lambda: toTestingPhase(controller))
        preset_button.grid(row=0, column=1, sticky="nsew")

        back_button = tk.Button(self, text="Go back", font=largeFont,
                                background=background_color_two, relief=RAISED, fg=text_color,
                                command=lambda: controller.show_frame(TrainingRoot.TrainingRoot))
        back_button.grid(row=1, column=0, columnspan=2, sticky="nsew")



