import tkinter as ttk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import joblib
import telepot

class App:
    def __init__(self, window, window_title, video_source):
        self.window = window
        self.window.title(window_title)

        # Open video source
        self.cap = cv2.VideoCapture(video_source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Create canvas for video frames
        self.canvas = ttk.Canvas(window, width=self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                                height=self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.photo = None
        self.update()

        window.geometry("1350x850")
        window.configure(background="#14161a")

        # Initialize Telegram bot
        self.bhavya = telepot.Bot("6258109822:AAEWVNpH7sUfzibhMdHEZctQZlYQ3ND7oHM")
        self.chatid_bhavya = "5425973951"

        # Load trained KNN model
        self.knn = joblib.load("knn.pkl")

        # GUI Labels and Entries
        label_title = ttk.Label(window, text='Cardiac Arrhythmia Prediction',
                                font=("Elephant", 24, "bold"),
                                background="#14161a", foreground="#ffffff")
        label_title.place(x=200, y=25)

        # Name
        ttk.Label(window, text='Name', font=("Times", 20, "bold"),
                  background="#14161a", foreground="#ffffff").place(x=100, y=150)
        self.Entry_0 = Entry(window, font=("Times", 16, "bold"), justify=CENTER)
        self.Entry_0.place(x=300, y=150)

        # Age
        ttk.Label(window, text='Age', font=("Times", 20, "bold"),
                  background="#14161a", foreground="#ffffff").place(x=100, y=200)
        self.Entry_1 = Entry(window, font=("Times", 16, "bold"), justify=CENTER)
        self.Entry_1.place(x=300, y=200)

        # Sex
        ttk.Label(window, text='Sex', font=("Times", 20, "bold"),
                  background="#14161a", foreground="#ffffff").place(x=100, y=250)
        self.options = StringVar(window)
        self.options.set("select option")
        self.om1 = OptionMenu(window, self.options, "Male", "Female")
        self.om1.place(x=300, y=250)

        # Height
        ttk.Label(window, text='Height (in cms)', font=("Times", 20, "bold"),
                  background="#14161a", foreground="#ffffff").place(x=100, y=300)
        self.Entry_3 = Entry(window, font=("Times", 16, "bold"), justify=CENTER)
        self.Entry_3.place(x=300, y=300)

        # Weight
        ttk.Label(window, text='Weight', font=("Times", 20, "bold"),
                  background="#14161a", foreground="#ffffff").place(x=100, y=350)
        self.Entry_4 = Entry(window, font=("Times", 16, "bold"), justify=CENTER)
        self.Entry_4.place(x=300, y=350)

        # QRS interval
        ttk.Label(window, text='QRS interval', font=("Times", 20, "bold"),
                  background="#14161a", foreground="#ffffff").place(x=100, y=400)
        self.Entry_5 = Entry(window, font=("Times", 16, "bold"), justify=CENTER)
        self.Entry_5.place(x=300, y=400)

        # PR interval
        ttk.Label(window, text='PR interval', font=("Times", 20, "bold"),
                  background="#14161a", foreground="#ffffff").place(x=100, y=450)
        self.Entry_6 = Entry(window, font=("Times", 16, "bold"), justify=CENTER)
        self.Entry_6.place(x=300, y=450)

        # T interval
        ttk.Label(window, text='T interval', font=("Times", 20, "bold"),
                  background="#14161a", foreground="#ffffff").place(x=100, y=500)
        self.Entry_7 = Entry(window, font=("Times", 16, "bold"), justify=CENTER)
        self.Entry_7.place(x=300, y=500)

        # Output label
        self.output = Label(window, font=("Times", 16, "bold"), background="#14161a", foreground="#ffffff")
        self.output.place(x=280, y=600)

        # Predict button
        b1 = Button(window, text='Predict', font=("Times", 20, "bold"),
                    background="#14161a", command=self.predict, foreground="#ffffff")
        b1.place(x=80, y=600)

        self.window.mainloop()

    def update(self):
        # Get a frame from the video source
        ret, frame = self.cap.read()
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=ttk.NW)

        # Repeat after 15 milliseconds
        self.window.after(15, self.update)

    def predict(self):
        try:
            name = self.Entry_0.get()
            age = self.Entry_1.get()
            sex = self.options.get()

            if sex == "Male":
                sex_id = 0
            elif sex == "Female":
                sex_id = 1
            else:
                self.output.configure(text="Please select sex", justify=CENTER)
                return

            height = self.Entry_3.get()
            weight = self.Entry_4.get()
            qrs_duration = self.Entry_5.get()
            p_r_interval = self.Entry_6.get()
            t_interval = self.Entry_7.get()

            # Convert inputs to float
            features = [
                float(age), float(sex_id), float(height), float(weight),
                float(qrs_duration), float(p_r_interval), float(t_interval)
            ]

            print("Features for prediction:", features)  # Debug print

            out = self.knn.predict([features])

            mapping = {
                1: 'Normal',
                2: 'Ischemic changes (Coronary Artery)',
                3: 'Old Anterior Myocardial Infarction',
                4: 'Old Inferior Myocardial Infarction',
                5: 'Sinus tachycardia',
                6: 'Ventricular Premature Contraction (PVC)',
                7: 'Supraventricular Premature Contraction',
                8: 'Left bundle branch block',
                9: 'Right bundle branch block',
                10: 'Left ventricle hypertrophy',
                11: 'Atrial Fibrillation',
                12: 'Atrial Flutter',
                13: 'Other Arrhythmias'
            }

            res = mapping.get(out[0], "Unknown")

            self.output.configure(text=res, justify=CENTER, font=("Times", 16, "bold"))
            self.bhavya.sendMessage(self.chatid_bhavya,
                                   f"Patient : {name}\nAge : {age}\nGender : {sex}\nHeight : {height}\nWeight : {weight}\nStatus : {res}")

            label_66 = ttk.Label(self.window, text=out[0], font=("Times", 20, "bold"),
                                 background="#14161a", foreground="#ffffff")
            label_66.place(x=400, y=650)

        except Exception as e:
            self.output.configure(text=f"Error: {e}", justify=CENTER, font=("Times", 16, "bold"))

# Run the app
App(ttk.Tk(), "Tkinter Video Looping Background", "video.mp4")
