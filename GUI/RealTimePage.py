import tkinter as tk
from tkinter import *

import TrainFilePage
import TrainStreamPage
import TrainingRoot

largeFont = ("Verdana", 12)
normalFont = ("Verdana", 10)


class RealTimePage(tk.Frame):
    """
    Page for branching to real time training with a data stream or prerecorded files.
    """
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.widgets(controller)

    def widgets(self, controller):
        background_color_one = "#8A9097"
        background_color_two = "#575F6B"
        text_color = "#FFFFFF"

        train_stream_button = tk.Button(self, text="Use data stream", font=largeFont,
                                        background=background_color_one, relief=GROOVE, fg=text_color,
                                        command=lambda: controller.show_frame(TrainStreamPage.TrainStreamPage))
        train_stream_button.pack(fill=BOTH, expand=True)

        train_file_button = tk.Button(self, text="Use data files", font=largeFont,
                                      background=background_color_one, relief=GROOVE, fg=text_color,
                                      command=lambda: controller.show_frame(TrainFilePage.TrainFilePage))
        train_file_button.pack(fill=BOTH, expand=True)

        back_button = tk.Button(self, text="Go back", font=largeFont,
                                background=background_color_two, relief=RAISED, fg=text_color,
                                command=lambda: controller.show_frame(TrainingRoot.TrainingRoot))
        back_button.pack(fill=BOTH, expand=True)
