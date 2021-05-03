import PIL
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image
import tkinter as tk
from tkinter import *


import GUIFunctions
import RealTimePage
import PresetPage
import ResearcherPage
import StartPage
import TrainingPage
import TrainingRoot
import UserPage

largeFont = ("Verdana", 12)
normalFont = ("Verdana", 10)


class DecisionPage(tk.Frame):
    """
    Page for deciding if the trained decoder is fit for continuing to the testing phase.
    """
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.widgets(controller)

    def widgets(self, controller):

        # GUI color scheme.
        background_color_one = "#8A9097"
        background_color_two = "#575F6B"
        text_color = "#FFFFFF"

        #image = PIL.Image.open("photo/Featurplot.png", "r")
        #image.resize((500, 500))
        #figure_eeg = Figure(figsize=(1, 1), dpi=100)
        #figure_eeg.figimage(image)
        #canvas_eeg = FigureCanvasTkAgg(figure_eeg, self)
        #canvas_eeg.draw()
        #canvas_eeg.get_tk_widget().pack()
        pass
