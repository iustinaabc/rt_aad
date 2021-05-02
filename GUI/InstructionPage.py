import tkinter as tk
from tkinter import *

import UserPage

largeFont = ("Verdana", 12)
normalFont = ("Verdana", 10)


class InstructionPage(tk.Frame):
    """
    Page for setting up user instructions.
    """
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.instruction_var = tk.StringVar()
        self.instruction_display = None
        self.instructions = ""
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=0)
        self.columnconfigure(3, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)
        self.widgets(controller)

    def widgets(self, controller):

        # GUI's color scheme.
        background_color_one = "#8A9097"
        background_color_two = "#575F6B"
        text_color = "#FFFFFF"

        left_entry = tk.Entry(self, font=largeFont,
                              background=background_color_one, relief=RAISED, fg=text_color)
        left_entry.grid(row=0, column=2, sticky="nsew")
        left_button = tk.Button(self, text="Add seconds left", font=largeFont,
                                background=background_color_two, relief=RAISED, fg=text_color,
                                command=lambda: self.addInstructionLeft(left_entry.get()))
        left_button.grid(row=0, column=3, sticky="nsew")

        right_entry = tk.Entry(self, font=largeFont,
                               background=background_color_one, relief=RAISED, fg=text_color)
        right_entry.grid(row=1, column=2, sticky="nsew")
        right_button = tk.Button(self, text="Add seconds right", font=largeFont,
                                 background=background_color_two, relief=RAISED, fg=text_color,
                                 command=lambda: self.addInstructionRight(right_entry.get()))
        right_button.grid(row=1, column=3, sticky="nsew")

        pause_entry = tk.Entry(self, font=largeFont,
                               background=background_color_one, relief=RAISED, fg=text_color)
        pause_entry.grid(row=2, column=2, sticky="nsew")
        pause_button = tk.Button(self, text="Add seconds to pause", font=largeFont,
                                 background=background_color_two, relief=RAISED, fg=text_color,
                                 command=lambda: self.addInstructionPause(pause_entry.get()))
        pause_button.grid(row=2, column=3, sticky="nsew")

        self.instruction_display = tk.Label(self, textvariable=self.instruction_var)
        self.instruction_display.grid(row=0, rowspan=3, column=0, columnspan=2, sticky="nsew")

        start_training = tk.Button(self, text="Continue", background=background_color_two,
                                   command=lambda: controller.show_frame(UserPage.UserPage, self.instructions))
        start_training.grid(row=3, column=0, columnspan=2, sticky="nsew")

        remove_button = tk.Button(self, text="Undo last instruction", background=background_color_two,
                                  command=lambda: self.removeInstruction())
        remove_button.grid(row=3, column=2, columnspan=2, sticky="nsew")

    def addInstructionLeft(self, seconds):
        """
        Add an instruction for listening to the left speaker.
        ----------
        seconds : int
            Amount of seconds to listen for.
        """
        text = self.instruction_var.get()
        updated_text = text + "\n Focus left for " + str(seconds) + " seconds."
        self.instruction_var.set(updated_text)
        self.instructions += (" left " + str(seconds))

    def addInstructionRight(self, seconds):
        """
        Add an instruction for listening to the right speaker.
        ----------
        seconds : int
            Amount of seconds to listen for.
        """
        text = self.instruction_var.get()
        updated_text = text + "\n Focus right for " + str(seconds) + " seconds."
        self.instruction_var.set(updated_text)
        self.instructions += (" right " + str(seconds))

    def addInstructionPause(self, seconds):
        """
        Add an instruction for pausing.
        ----------
        seconds : int
            Amount of seconds to pause for.
        """
        text = self.instruction_var.get()
        updated_text = text + "\n Pause now for " + str(seconds) + " seconds."
        self.instruction_var.set(updated_text)
        self.instructions += (" pause " + str(seconds))

    def removeInstruction(self):
        """
        Remove the last instruction from the display and the class instruction parameter.
        """
        self.instructions = " ".join(self.instructions.split()[:-2])

        splits = self.instruction_var.get().split()[:-5]
        i = 0
        while i < len(splits):
            splits.insert(i, "\n")
            i += 6
        self.instruction_var.set(" ".join(splits))
