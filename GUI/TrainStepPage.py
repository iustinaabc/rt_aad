import tkinter as tk
from tkinter import *

import RealTimePage
import main

largeFont = ("Verdana", 12)
normalFont = ("Verdana", 10)


class TrainStepPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.parameter_var = tk.StringVar()
        self.parameter_display = None
        self.update_entry = None
        self.widgets(controller)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=0)
        self.columnconfigure(3, weight=0)
        self.args = None

    def widgets(self, controller):

        background_color_one = "#8A9097"
        background_color_two = "#575F6B"
        text_color = "#FFFFFF"

        self.parameter_display = tk.Label(self, textvariable=self.parameter_var)
        self.parameter_display.grid(row=0, rowspan=2, column=0, columnspan=2, sticky="nsew")

        parameter_button = tk.Button(self, text="Show parameters", font=largeFont,
                                     background=background_color_two, relief=RAISED, fg=text_color,
                                     command=lambda: self.updateArgs())
        parameter_button.grid(row=0, column=2, sticky="nsew")
        self.initialize()

        update_frame = tk.Frame(self)
        update_frame.rowconfigure(0, weight=1)
        update_frame.grid(row=1, column=2, sticky="nsew")
        self.update_entry = tk.Entry(update_frame, font=largeFont,
                                     background=background_color_one, relief=GROOVE, fg=text_color)
        self.update_entry.grid(row=0, column=0, sticky="nsew")
        update_button = tk.Button(update_frame, text="Change parameter", font=largeFont,
                                  background=background_color_two, relief=RAISED, fg=text_color,
                                  command=lambda: self.changeParameter())
        update_button.grid(row=0, column=1, sticky="nsew")

        commit_button = tk.Button(self, text="Commit variables", font=largeFont,
                                  background=background_color_two, relief=RAISED, fg=text_color,
                                  command=lambda: self.train())
        commit_button.grid(row=0, column=3, sticky="nsew")

        back_button = tk.Button(self, text="Go back", font=largeFont,
                                background=background_color_two, relief=RAISED, fg=text_color,
                                command=lambda: controller.show_frame(RealTimePage.RealTimePage))
        back_button.grid(row=1, column=3, sticky="nsew")

    def initialize(self):
        """
        Show the parameters from the main function to the user.
        """
        dictionary = main.PARAMETERS
        parameters = ""
        for i in dictionary:
            parameters += str(i)
            parameters += " : "
            parameters += str(dictionary[i])
            parameters += "\n"
        self.parameter_var.set(parameters)

    def updateArgs(self):
        """
        Change NoTraining and RealtimeTraining parameters for main path determination.
        """
        if self.args is not None:
            self.changeParameter("NoTraining : " + str(self.args[0]))
            self.changeParameter("RealtimeTraining : " + str(self.args[1]))

    def changeParameter(self, args=None):
        """
        Change parameters according to the variable in the entry widget.
        """
        if args is not None:
            update = args
        else:
            update = self.update_entry.get()
        update_splits = update.split()
        parameters = self.parameter_var.get().splitlines()
        self.parameter_var.set("")
        for i in range(0, len(parameters)):
            if parameters[i].startswith(str(update_splits[0])):
                parameters[i] = str(update)
            text = self.parameter_var.get()
            text += str(parameters[i])
            text += " \n"
            self.parameter_var.set(text)

    def train(self):
        """
        Commence training with the given parameters.
        """
        parameters = self.parameter_var.get().split()
        dictionary = {}
        for i in range(0, len(parameters)-1):
            dictionary[parameters[i]] = parameters[i+1]
            i += 2
        main.main(dictionary)
