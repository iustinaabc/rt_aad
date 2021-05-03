import tkinter as tk
from tkinter import *

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

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


class ResearcherPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.widgets(controller)
        # Scaling of relative weights of rows and columns.
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)

    def graphing(self):

        # Left graph for real time plotting.
        image_feat = Image.open("photo/Featureplot.png")
        image_feat = image_feat.resize((450, 350), Image.ANTIALIAS)
        figure_feat = Figure(figsize=(1, 1), dpi=100)
        figure_feat.figimage(image_feat)
        canvas_feat = FigureCanvasTkAgg(figure_feat, self)
        canvas_feat.draw()
        canvas_feat.get_tk_widget().grid(row=1, column=1, sticky="nsew")

        # Right graph for feature plotting.
        image_EEG = Image.open("photo/Featureplot.png")
        image_EEG = image_EEG.resize((450, 350), Image.ANTIALIAS)
        figure_EEG = Figure(figsize=(1, 1), dpi=100)
        figure_EEG.figimage(image_EEG)
        canvas_EEG = FigureCanvasTkAgg(figure_EEG, self)
        canvas_EEG.draw()
        canvas_EEG.get_tk_widget().grid(row=1, column=2, sticky="nsew")

    def updateGraphing(self):
        pass

    def widgets(self, controller):

        backgroundcolorone = "#8A9097"
        backgroundcolortwo = "#575F6B"
        textcolor = "#FFFFFF"

        pageLabel = tk.Label(self, text="Researcher Page", font=largeFont, fg=textcolor,
                             background=backgroundcolortwo, relief=RAISED)
        pageLabel.grid(row=0, column=0, sticky="nsew")
        fillerLabel =tk.Label(self, background=backgroundcolortwo)
        fillerLabel.grid(row=1, column=0, sticky="nsew")

        researcherButton = tk.Button(self, text="User Page", font=largeFont, fg=textcolor,
                                     command=lambda: controller.show_frame(UserPage.UserPage),
                                     background=backgroundcolortwo, relief=RAISED)
        researcherButton.grid(row=0, column=1, sticky="nsew")

        homeButton = tk.Button(self, text="Start page", font=largeFont, fg=textcolor,
                               command=lambda: controller.show_frame(StartPage.StartPage),
                               background=backgroundcolortwo, relief=RAISED)
        homeButton.grid(row=0, column=2, sticky="nsew")

        graphingButton = tk.Button(self, font=largeFont,
                         text="graph", command=lambda: self.graphing(),
                         background=backgroundcolortwo, relief=RAISED)
        graphingButton.grid(row=1, column=0, sticky="nsew")