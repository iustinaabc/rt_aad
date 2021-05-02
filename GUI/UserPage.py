import tkinter as tk
from tkinter import *
import time

import ResearcherPage
import StartPage

largeFont = ("Verdana", 12)
normalFont = ("Verdana", 10)


class UserPage(tk.Frame):
    """
    Page that allows a test subject to follow instructions.
    """
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.audio_trainer = None
        self.widgets(controller)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)
        self.rowconfigure(3, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.args = None

    def widgets(self, controller):

        # GUI's color scheme.
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

        # Text display for user instructions.
        self.audio_trainer = tk.Text(self, width=40, height=10, background=background_color_one, fg=text_color)
        self.audio_trainer.insert(tk.END, "Your instructions will be displayed here.")
        self.audio_trainer.grid(row=1, rowspan=3, column=0, sticky="nsew")

        # Indicates the user attended to the right speaker and it's volume should be increased.
        def increase():
            volume.set(volume.get() + 1)

        # Indicates the user attended to the left speaker and it's volume should be increased.
        def decrease():
            volume.set(volume.get() - 1)

        # Volume slider.
        volume = tk.Scale(self, orient='horizontal', from_=-100, to_=100,
                          length=300, background=background_color_two)
        volume.grid(row=1, column=1, sticky="nsew")

        # Instruction input.
        training_frame = tk.Frame(self)
        training_frame.columnconfigure(0, weight=1)
        training_frame.columnconfigure(1, weight=0)
        training_frame.rowconfigure(0, weight=1)
        training_frame.grid(row=2, column=1, sticky="nsew")
        training_entry_widget = tk.Entry(training_frame, background=background_color_two)
        training_entry_widget.grid(row=0, column=0, sticky="nsew")
        training_button = tk.Button(training_frame, background=background_color_two,
                                    command=lambda: self.train())
        training_button.grid(row=0, column=1, sticky="nsew")

        # System's choice indicator.
        choice_frame = tk.Frame(self)
        choice_frame.grid(row=3, column=1, sticky="nsew")
        choice_var = IntVar()
        choice_one = tk.Radiobutton(choice_frame, text="Left", variable=choice_var, value=0)
        choice_one.pack(side=tk.LEFT, padx=50)
        choice_two = tk.Radiobutton(choice_frame, text="Right", variable=choice_var, value=1)
        choice_two.pack(side=tk.RIGHT, padx=50)

    def train(self):
        """
        Display the instructions from the entry widget to the text widget.
        ----------
        instruction_set : list of alternating strings and integers
            Inputs to the updateText() function.
        """
        instructions = self.args.split()
        if not len(instructions) % 2 == 0:
            self.updateText("Please input an even number of arguments", 0)
            return
        self.updateText("Training will start in 3...", 1)
        self.updateText("Training will start in 2...", 1)
        self.updateText("Training will start in 1...", 1)
        self.audio_trainer.delete('1.0', END)
        for i in range(0, len(instructions), 2):
            duration = int(instructions[i + 1])
            if instructions[i] == "left" or instructions[i] == "right":
                self.updateText("Please focus on the " + str(instructions[i]) + " speaker for " +
                                str(duration) + " seconds.", duration)
            else:
                self.updateText("Next instruction in " + str(duration) + " seconds.", duration)
            self.audio_trainer.delete('1.0', END)
        self.updateText("Training completed.", 0)

    def updateText(self, update, timer):
        """
        Update the page's text widget with the text from the update parameter and wait for the amount of
        seconds in the timer parameter.
        ----------
        update : str
            Instruction to append to the text widget.
        timer : int
            Amount of seconds to wait after the instruction.
        """
        self.audio_trainer.insert(tk.END, "\n")
        self.audio_trainer.insert(tk.END, update)
        self.audio_trainer.update()
        time.sleep(timer)
