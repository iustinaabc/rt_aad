import tkinter as tk

import TrainingRoot
import UserPage
import InstructionPage

largeFont = ("Verdana", 12)
normalFont = ("Verdana", 10)


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.widgets(controller)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=2)
        self.rowconfigure(2, weight=2)
        self.rowconfigure(3, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)

    def widgets(self, controller):

        background_color_one = "#8A9097"
        background_color_two = "#575F6B"
        text_color = "#FFFFFF"

        # Background filling labels.
        bg00 = tk.Label(self, background=background_color_one)
        bg00.grid(row=0, column=0, sticky="nsew")
        bg10 = tk.Label(self, background=background_color_one)
        bg10.grid(row=1, column=0, sticky="nsew")
        bg02 = tk.Label(self, background=background_color_one)
        bg02.grid(row=0, column=2, sticky="nsew")
        bg12 = tk.Label(self, background=background_color_one)
        bg12.grid(row=1, column=2, sticky="nsew")

        head_label = tk.Label(self, text="Real Time Audio Decoder", font=largeFont,
                              background=background_color_one, relief="groove")
        head_label.grid(row=0, column=1, sticky="nsew")

        welcome_text = tk.Label(self, text="Welcome to the interface's start page! \r\n"
                                           "Navigate to the training page to set up"
                                           "a new controller or load a preset. \r\n",
                                font=normalFont, background=background_color_one)
        welcome_text.grid(row=1, column=0, columnspan=3, sticky="nsew")

        user_button = tk.Button(self, text="USER PAGE \r\n"
                                           "user instructions and stereo user interface",
                                command=lambda: controller.show_frame(UserPage.UserPage),
                                background=background_color_two, fg=text_color)
        user_button.grid(row=2, column=0, sticky="nsew")

        researcher_button = tk.Button(self, text="INSTRUCTION PAGE \r\n"
                                                 "set up user instructions for training and testing",
                                      command=lambda: controller.show_frame(InstructionPage.InstructionPage),
                                      background=background_color_two, fg=text_color)
        researcher_button.grid(row=2, column=1, sticky="nsew")

        training_button = tk.Button(self, text="TRAINING PAGE \r\n"
                                               "configure training setup for new controllers",
                                    command=lambda: controller.show_frame(TrainingRoot.TrainingRoot),
                                    background=background_color_two, fg=text_color)
        training_button.grid(row=2, column=2, sticky="nsew")

        creator_label = tk.Label(self, text="Developed by Bas, Nele and Sofie")
        creator_label.grid(row=3, column=1, sticky="nsew")
