import tkinter as tk
from tkinter import *

import RealTimePage
import PresetPage
import StartPage

largeFont = ("Verdana", 12)
normalFont = ("Verdana", 10)


class TrainingRoot(tk.Frame):
    """
    Central page for branching to the different possibilities for training a decoder.
    """
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.widgets(controller)

    def widgets(self, controller):

        background_color_one = "#8A9097"
        background_color_two = "#575F6B"
        text_color = "#FFFFFF"

        real_time_button = tk.Button(self, text="Training", font=largeFont,
                                     background=background_color_one, relief=GROOVE, fg=text_color,
                                     command=lambda: controller.show_frame(RealTimePage.RealTimePage))
        real_time_button.pack(fill=BOTH, expand=True)

        preset_button = tk.Button(self, text="No training (preprocessed filters)", font=largeFont,
                                  background=background_color_one, relief=GROOVE, fg=text_color,
                                  command=lambda: controller.show_frame(PresetPage.PresetPage))
        preset_button.pack(fill=BOTH, expand=True)

        back_button = tk.Button(self, text="Go back", font=largeFont,
                                background=background_color_two, relief=RAISED, fg=text_color,
                                command=lambda: controller.show_frame(StartPage.StartPage))
        back_button.pack(fill=BOTH, expand=True)
