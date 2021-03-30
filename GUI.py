import matplotlib
import tkinter as tk
from tkinter import *
import tkinter.filedialog
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter import ttk

matplotlib.use("TkAgg")

largeFont = ("Verdana", 12)
normalFont = ("Verdana", 10)


# Display popups as mini tkinter instances, only destroyed on user interaction.
def popupMessage(message):
    # Destroy popups.
    def leaveMini():
        popup.destroy()

    width = 400
    height = 100
    popup = tk.Tk()
    popup.wm_title("!")
    screenwidth = popup.winfo_screenwidth()
    screenheight = popup.winfo_screenheight()
    x = (screenwidth / 2) - (width / 2)
    y = (screenheight / 2) - (height / 2)
    popup.geometry('%dx%d+%d+%d' % (width, height, x, y))
    label = ttk.Label(popup, text=message, font=normalFont)
    label.pack(side="top", fill="x", pady=10)
    button1 = ttk.Button(popup, text="Okay", command=leaveMini)
    button1.pack()
    popup.mainloop()


# Popup that destroys itself after the given duration.
def selfDestructMessage(message, duration):
    width = 400
    height = 100
    popup = tk.Tk()
    screenwidth = popup.winfo_screenwidth()
    screenheight = popup.winfo_screenheight()
    x = (screenwidth / 2) - (width / 2)
    y = (screenheight / 2) - (height / 2)
    popup.geometry('%dx%d+%d+%d' % (width, height, x, y))
    popup.wm_title("!")
    label = ttk.Label(popup, text=message, font=normalFont)
    label.pack(side="top", fill="x", pady=10)
    popup.after(duration * 1000, lambda: popup.destroy())
    popup.mainloop()


# GUIMain inherits from tkinter.
class GUIMain(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.wm_title(self, "PENO4")

        # Window initialisation.
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        # Minimum size of zero, both with equal priority weights.
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Navigation widget
        menubar = tk.Menu(container)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Save settings", command=lambda: popupMessage("not yet"))
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=quit)
        menubar.add_cascade(label="File", menu=filemenu)
        tk.Tk.config(self, menu=menubar)

        # Tuple containing all pages.
        self.frames = {}
        for F in (StartPage, UserPage, ResearcherPage, TrainingPage):
            frame = F(container, self)
            self.frames[F] = frame

            # Create grid-like structure.
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    # Load page to front corresponding to key value in controller.
    def show_frame(self, controller):
        frame = self.frames[controller]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Start Page", font=largeFont)
        label.pack(pady=10, padx=10)

        userButton = tk.Button(self, text="User page",
                               command=lambda: controller.show_frame(UserPage))
        userButton.pack()

        researcherButton = tk.Button(self, text="Researcher page",
                                     command=lambda: controller.show_frame(ResearcherPage))
        researcherButton.pack()

        trainingButton = tk.Button(self, text="Training page",
                                   command=lambda: controller.show_frame(TrainingPage))
        trainingButton.pack()


# Page for training models based on (pre)recorded data.
class TrainingPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.widgets(controller)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)
        self.rowconfigure(3, weight=1)
        self.rowconfigure(4, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self.columnconfigure(3, weight=1)
        self.columnconfigure(4, weight=1)
        self.columnconfigure(5, weight=1)

    """ Ask the user for a file and store it as a variable. """
    @staticmethod
    def askDirectory(self, var):
        direc = str(tk.filedialog.askdirectory()).replace("/", "\\")
        if direc:
            var.set(direc)

    """ Convert binary boolean values to actual boolean values. """
    def widgets(self, controller):
        def trueFalse(var):
            if var == 0:
                return False
            else:
                return True

        # Read user input values.
        def collectParameters():
            data = {}
            datatype = datatypeEntry.get()
            print("datatype: " + datatype)
            data.__setitem__("datatype", datatype)
            samplingFrequency = samplingFrequencyEntry.get()
            print("samplingFrequency: " + samplingFrequency)
            data.__setitem__("samplingFrequency", samplingFrequency)
            channels = channelsEntry.get()
            print("channels: " + channels)
            data.__setitem__("channels", channels)
            timeframe = timeframeEntry.get()
            print("timeframe: " + timeframe)
            data.__setitem__("timeframe", timeframe)
            overlap = overlapEntry.get()
            print("overlap: " + overlap)
            data.__setitem__("overlap", overlap)
            trainingDataset = trainingDatasetVar.get()
            print("trainingDataset: " + str(trainingDataset))
            data.__setitem__("trainingDataset", trainingDataset)
            updateCSP = trueFalse(CSPCheck.get())
            print("updateCSP: " + str(updateCSP))
            data.__setitem__("updateCSP", updateCSP)
            updateCov = trueFalse(CovCheck.get())
            print("updateCov: " + str(updateCov))
            data.__setitem__("updateCov", updateCov)
            updateBias = trueFalse(BiasCheck.get())
            print("updateBias: " + str(updateBias))
            data.__setitem__("updateBias", updateBias)
            windowLengthTraining = windowLengthTrainingEntry.get()
            print("windowLengthTraining: " + windowLengthTraining)
            data.__setitem__("windowLengthTraining", windowLengthTraining)
            location_eeg1 = location_eeg1Var.get()
            print("location_eeg1: " + str(location_eeg1))
            data.__setitem__("location_eeg1", location_eeg1)
            location_eeg2 = location_eeg2Var.get()
            print("location_eeg2: " + str(location_eeg2))
            data.__setitem__("location_eeg2", location_eeg2)
            dumpTrainingData = trueFalse(dumpTrainingDataVar.get())
            print("dumpTrainingData: " + str(dumpTrainingData))
            data.__setitem__("dumpTrainingData", dumpTrainingData)
            print("\r\n")
            print(data)

            #mainParameters = [datatype, samplingFrequency, channels, timeframe, overlap, trainingDataset, updateCSP,
            #                  updateCov, updateBias, windowLengthTraining, location_eeg1, location_eeg2,
            #                  dumpTrainingData, None, None]

            return data

        datatypeLabel = tk.Label(self, text="datatype:")
        datatypeLabel.grid(row=0, column=0, sticky="nsew")
        datatypeEntry = tk.Entry(self)
        datatypeEntry.grid(row=0, column=1, sticky="nsew")
        samplingFrequencyLabel = tk.Label(self, text="samplingFrequency:")
        samplingFrequencyLabel.grid(row=0, column=2, sticky="nsew")
        samplingFrequencyEntry = tk.Entry(self)
        samplingFrequencyEntry.grid(row=0, column=3, sticky="nsew")
        channelsLabel = tk.Label(self, text="channels:")
        channelsLabel.grid(row=0, column=4, sticky="nsew")
        channelsEntry = tk.Entry(self)
        channelsEntry.grid(row=0, column=5, sticky="nsew")
        timeframeLabel = tk.Label(self, text="timeframe:")
        timeframeLabel.grid(row=1, column=0, sticky="nsew")
        timeframeEntry = tk.Entry(self)
        timeframeEntry.grid(row=1, column=1, sticky="nsew")
        overlapLabel = tk.Label(self, text="overlap:")
        overlapLabel.grid(row=1, column=2, sticky="nsew")
        overlapEntry = tk.Entry(self)
        overlapEntry.grid(row=1, column=3, sticky="nsew")
        trainingDatasetLabel = tk.Label(self, text="trainingDataset:")
        trainingDatasetLabel.grid(row=1, column=4, sticky="nsew")
        trainingDatasetVar = StringVar()
        trainingDatasetButton = tk.Button(self, text='Choose directory',
                                          command=lambda: TrainingPage.askDirectory(self, trainingDatasetVar))
        trainingDatasetButton.grid(row=1, column=5, sticky="nsew")
        updateCSPLabel = tk.Label(self, text="updateCSP?")
        updateCSPLabel.grid(row=2, column=0, sticky="nsew")
        CSPCheck = tk.IntVar()
        updateCSPCheck = tk.Checkbutton(self, variable=CSPCheck)
        updateCSPCheck.grid(row=2, column=1, sticky="nsew")
        updateCovLabel = tk.Label(self, text="updateCov?")
        updateCovLabel.grid(row=2, column=2, sticky="nsew")
        CovCheck = tk.IntVar()
        updateCovCheck = tk.Checkbutton(self, variable=CovCheck)
        updateCovCheck.grid(row=2, column=3, sticky="nsew")
        updateBiasLabel = tk.Label(self, text="updateBias?")
        updateBiasLabel.grid(row=2, column=4, sticky="nsew")
        BiasCheck = tk.IntVar()
        updateBiasCheck = tk.Checkbutton(self, variable=BiasCheck)
        updateBiasCheck.grid(row=2, column=5, sticky="nsew")
        windowLengthTrainingLabel = tk.Label(self, text="windowLengthTraining:")
        windowLengthTrainingLabel.grid(row=3, column=0, sticky="nsew")
        windowLengthTrainingEntry = tk.Entry(self)
        windowLengthTrainingEntry.grid(row=3, column=1, sticky="nsew")
        location_eeg1Label = tk.Label(self, text="EEG1 location:")
        location_eeg1Label.grid(row=3, column=2, sticky="nsew")
        location_eeg1Var = StringVar()
        location_eeg1Button = tk.Button(self, text='EEG1',
                                        command=lambda: TrainingPage.askDirectory(self, location_eeg1Var))
        location_eeg1Button.grid(row=3, column=3, sticky="nsew")
        location_eeg2Label = tk.Label(self, text="EEG2 location:")
        location_eeg2Label.grid(row=3, column=4, sticky="nsew")
        location_eeg2Var = StringVar()
        location_eeg2Button = tk.Button(self, text='EEG2',
                                        command=lambda: TrainingPage.askDirectory(self, location_eeg2Var))
        location_eeg2Button.grid(row=3, column=5, sticky="nsew")
        dumpTrainingDataLabel = tk.Label(self, text="dumpTrainingData?")
        dumpTrainingDataLabel.grid(row=4, column=0, sticky="nsew")
        dumpTrainingDataVar = tk.IntVar()
        dumpTrainingDataCheck = tk.Checkbutton(self, variable=dumpTrainingDataVar)
        dumpTrainingDataCheck.grid(row=4, column=1, sticky="nsew")
        tempLabel1 = tk.Label(self, text="???")
        tempLabel1.grid(row=4, column=2, sticky="nsew")
        tempButton1 = tk.Button(self)
        tempButton1.grid(row=4, column=3, sticky="nsew")
        startLabel = tk.Label(self, text="Start training:")
        startLabel.grid(row=4, column=4, sticky="nsew")
        startButton = tk.Button(self, text="START", command=lambda: collectParameters())
        startButton.grid(row=4, column=5, sticky="nsew")


class UserPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.widgets(controller)

    def widgets(self, controller):
        label = tk.Label(self, text="User Page", font=largeFont)
        label.grid(row=0, column=0, sticky="nsew")

        # Navigation buttons for other pages.
        userButton = tk.Button(self, text="Researcher page",
                               command=lambda: controller.show_frame(ResearcherPage))
        userButton.grid(row=0, column=1, sticky="nsew")

        homeButton = tk.Button(self, text="Start page",
                               command=lambda: controller.show_frame(StartPage))
        homeButton.grid(row=0, column=3, sticky="nsew")

        # Indicates the user attended to the right speaker and it's volume should be increased.
        def increase():
            value = volume.get()
            volume.set(value + 1)
            btn_increase['image'] = speaker_on
            btn_decrease['image'] = speaker_off

        # Indicates the user attended to the left speaker and it's volume should be increased.
        def decrease():
            value = volume.get()
            volume.set(value - 1)
            btn_decrease['image'] = speaker_on
            btn_increase['image'] = speaker_off

        # The speaker icons used for the speaker buttons.
        speaker_off = tk.PhotoImage(
            file=r'C:\Users\gebruiker\Documents\KULeuven\2020-2021\PENO\Speaker_Highlighted.png')
        speaker_on = tk.PhotoImage(
            file=r'C:\Users\gebruiker\Documents\KULeuven\2020-2021\PENO\Speaker_Unhighlighted.png')

        # Right speaker button.
        btn_decrease = tk.Button(self, image=speaker_off, command=decrease)
        btn_decrease.grid(row=1, column=0, sticky="nsew")

        # Volume slider.
        volume = tk.Scale(self, orient='horizontal',
                          from_=-100, to_=100, activebackground='red',
                          length=300)
        volume.grid(row=0, column=2)

        # Left speaker button.
        btn_increase = tk.Button(self, image=speaker_off, command=increase)
        btn_increase.grid(row=1, column=1, sticky="nsew")

        audio_trainer = tk.Text(self, width=40, height=10, bd=5)
        audio_trainer.insert(tk.END,
                             "Display subject instructions here")
        audio_trainer.grid(row=1, column=2)


class ResearcherPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.widgets(controller)
        # Scaling of relative weights of rows and columns.
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=5)
        self.rowconfigure(2, weight=5)
        self.columnconfigure(1, weight=3)
        self.columnconfigure(0, weight=15)

    def widgets(self, controller):

        label = tk.Label(self, text="Researcher Page", font=largeFont)
        label.grid(row=0, column=0, sticky="nsew")

        researcherButton = tk.Button(self, text="User Page",
                                     command=lambda: controller.show_frame(UserPage))
        researcherButton.grid(row=0, column=1, sticky="nsew")

        homeButton = tk.Button(self, text="Start page",
                               command=lambda: controller.show_frame(StartPage))
        homeButton.grid(row=0, column=2, sticky="nsew")

        f = Figure(figsize=(5, 5), dpi=100)
        sub = f.add_subplot(111)
        sub.plot([1, 2, 3, 4], [5, 6, 7, 8])

        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=2, column=0, sticky="nsew")

        """ Set up instructions for the user in repeating LEFT/RIGHT TIME format. """
        def train():
            program = trainingEntryWidget.get()
            program = program.split()
            if not len(program) % 2 == 0:
                print("Please input an even number of arguments")
                return
            for i in range(0, len(program), 2):
                # Display popup with user command.
                duration = int(program[i + 1])
                selfDestructMessage("Please focus " + str(program[i]) + " for " + str(program[i + 1]) + " seconds",
                                    duration)

        trainingEntryWidget = tk.Entry(self)
        trainingEntryWidget.grid(row=1, column=1, sticky="nsew")

        trainingButton = tk.Button(self, text="Start Training", command=train)
        trainingButton.grid(row=1, column=2, sticky="nsew")


app = GUIMain()
app.mainloop()
