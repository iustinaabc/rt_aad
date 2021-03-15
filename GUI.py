import matplotlib
import tkinter as tk
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
    x = (screenwidth/2) - (width/2)
    y = (screenheight/2) - (height/2)
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
    x = (screenwidth/2) - (width/2)
    y = (screenheight/2) - (height/2)
    popup.geometry('%dx%d+%d+%d' % (width, height, x, y))
    popup.wm_title("!")
    label = ttk.Label(popup, text=message, font=normalFont)
    label.pack(side="top", fill="x", pady=10)
    popup.after(duration*1000, lambda: popup.destroy())
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
        for F in (StartPage, UserPage, ResearcherPage):
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
        canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        # Set up instructions for the user in repeating LEFT/RIGHT TIME format.
        def train():
            program = trainingEntryWidget.get()
            program = program.split()
            if not len(program) % 2 == 0:
                print("Please input an even number of arguments")
                Return
            for i in range(0, len(program), 2):
                # Display popup with user command.
                duration = int(program[i+1])
                selfDestructMessage("Please focus " + str(program[i]) + " for " + str(program[i+1]) + " seconds",
                                    duration)

        trainingEntryWidget = tk.Entry(self)
        trainingEntryWidget.grid(row=1, column=1, sticky="nsew")

        trainingButton = tk.Button(self, text="Start Training", command=train)
        trainingButton.grid(row=1, column=2, sticky="nsew")


app = GUIMain()
app.mainloop()