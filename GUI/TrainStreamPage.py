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


class TrainStreamPage(tk.Frame):
    """
    Page that...
    """
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.widgets(controller)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)
        self.rowconfigure(3, weight=1)
        self.rowconfigure(4, weight=0)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

    def widgets(self, controller):

        background_color_one = "#8A9097"
        background_color_two = "#575F6B"
        text_color = "#FFFFFF"

        filter_band_label = tk.Label(self, text="Filter Band:", font=largeFont,
                                     background=background_color_two, relief=RAISED, fg=text_color)
        filter_band_label.grid(row=0, column=0, sticky="nsew")
        filter_frame = tk.Frame(self)
        filter_frame.grid(row=0, column=1, sticky="nsew")
        filter_band_bank_low = tk.Scale(filter_frame, from_=0, to=60, background=background_color_one, orient="horizontal")
        filter_band_bank_low.pack(fill=BOTH, expand=True)
        filter_band_bank_low.set(12)
        filter_band_bank_high = tk.Scale(filter_frame, from_=0, to=60, background=background_color_one, orient="horizontal")
        filter_band_bank_high.pack(fill=BOTH, expand=True)
        filter_band_bank_high.set(30)

        sampling_frequency_label = tk.Label(self, text="Sampling Frequency", font=largeFont,
                                            background=background_color_two, relief=RAISED, fg=text_color)
        sampling_frequency_label.grid(row=0, column=2, sticky="nsew")
        sampling_frequency_entry = tk.Entry(self, font=largeFont,
                                            background=background_color_one, relief=GROOVE)
        sampling_frequency_entry.insert(END, 120)
        sampling_frequency_entry.grid(row=0, column=3, sticky="nsew")

        decision_window_label = tk.Label(self, text="samplingFrequency:", font=largeFont,
                                         background=background_color_two, relief=RAISED, fg=text_color)
        decision_window_label.grid(row=1, column=0, sticky="nsew")
        decision_window_entry = tk.Entry(self, font=largeFont,
                                         background=background_color_one, relief=GROOVE)
        decision_window_entry.insert(END, 10)
        decision_window_entry.grid(row=1, column=1, sticky="nsew")

        save_training_label = tk.Label(self, text="Save training data?", font=largeFont,
                                       background=background_color_two, relief=RAISED, fg=text_color)
        save_training_label.grid(row=1, column=2, sticky="nsew")
        training_var = StringVar()
        save_training_button = tk.Button(self, text="Choose save location", font=largeFont,
                                         background=background_color_one, relief=GROOVE,
                                         command=lambda: TrainingPage.askDirectory(self, training_var))
        save_training_button.grid(row=1, column=3, sticky="nsew")

        channels_label = tk.Label(self, text="Channels", font=largeFont,
                                  background=background_color_two, relief=RAISED, fg=text_color)
        channels_label.grid(row=2, column=0, sticky="nsew")
        channels_entry = tk.Entry(self, font=largeFont,
                                  background=background_color_one, relief=GROOVE)
        channels_entry.insert(END, 24)
        channels_entry.grid(row=2, column=1, sticky="nsew")

        EEG_sampling_frequency_label = tk.Label(self, text="Sampling Frequency", font=largeFont,
                                                background=background_color_two, relief=RAISED, fg=text_color)
        EEG_sampling_frequency_label.grid(row=2, column=2, sticky="nsew")
        EEG_sampling_frequency_entry = tk.Entry(self, font=largeFont,
                                                background=background_color_one, relief=GROOVE)
        EEG_sampling_frequency_entry.insert(END, 120)
        EEG_sampling_frequency_entry.grid(row=2, column=3, sticky="nsew")

        back_button = tk.Button(self, text="Go back", font=largeFont,
                                background=background_color_two, relief=RAISED, fg=text_color,
                                command=lambda: controller.show_frame(RealTimePage.RealTimePage))
        back_button.grid(row=3, columnspan=4, column=0, sticky="nsew")

        start_button = tk.Button(self, text="Start", font=largeFont,
                                 background=background_color_two, relief=RAISED,
                                 command=lambda: self.trainStream())
        start_button.grid(row=4, column=0, columnspan=4, sticky="nsew")

    def trainStream(self):
        train_again = True
        while train_again:
            GUIFunctions.popupMessage("Training filters and LDA")


