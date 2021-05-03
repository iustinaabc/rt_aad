import tkinter as tk

import RealTimePage
import PresetPage
import ResearcherPage
import StartPage
import TrainStepPage
import TrainingPage
import TrainingRoot
import UserPage
import DecisionPage
import InstructionPage
import TestingPage

largeFont = ("Verdana", 12)
normalFont = ("Verdana", 10)


class GUIMain(tk.Tk):
    """
    Starting page for the GUI.
    """
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.wm_title(self, "Projectwerk ontwerpen in de biomedische technologie")

        # Window initialisation.
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        # Minimum size of zero, both with equal priority weights.
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Navigation widget
        menu_bar = tk.Menu(container)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Exit", command=quit)
        menu_bar.add_cascade(label="Options", menu=file_menu)
        tk.Tk.config(self, menu=menu_bar)

        # Tuple containing all pages.
        self.frames = {}
        for F in (StartPage.StartPage, UserPage.UserPage, ResearcherPage.ResearcherPage,
                  TrainingPage.TrainingPage, TrainingRoot.TrainingRoot,
                  RealTimePage.RealTimePage, TrainStepPage.TrainStepPage,
                  PresetPage.PresetPage, DecisionPage.DecisionPage,
                  InstructionPage.InstructionPage, TestingPage.TestingPage):

            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage.StartPage)

    def show_frame(self, controller, args=None):
        """
        Show the given page to the user.
        ----------
        controller : GUI class to show
            Class Page to display.
        args : Any
            Parameters to be transferred when switching pages.
        """
        frame = self.frames[controller]
        frame.tkraise()
        if args is not None:
            frame.args = args


# Default root window
if __name__ == "__main__":
    app = GUIMain()
    app.mainloop()
