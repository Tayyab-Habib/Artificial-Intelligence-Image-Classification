#importing some libraries
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def newWindow():
    from tensorflow.keras import datasets, layers, models
    master.destroy()
    (training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
    training_images, testing_images = training_images / 255, testing_images / 255

    class_name = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

    training_images = training_images[:20000]
    training_labels = training_labels[:20000]
    testing_images = testing_images[:4000]
    testing_labels = testing_labels[:4000]

    model = models.load_model('image_classifier.model')

    ##GUI STUFF
    def Upload_Image():
        file_Path = filedialog.askopenfilename()  # Show a Pop-Up from which image can be selected
        uploaded = Image.open(file_Path)  # Store the file
        uploaded.thumbnail(((front.winfo_width() / 2.25), (front.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text=' ')  # Where predicted word will be shown
        show_classify_button(file_Path)  # WHen the image is uploaded then to show the classification button to classify

    def show_classify_button(file_path):
        classify_btn = Button(front, text="Click to Classify", command=lambda: classify(file_path), padx=10, pady=5)
        classify_btn.configure(background="#364156", foreground="white", font=('arial', 14, 'bold'))
        classify_btn.place(relx=0.38, rely=0.70)

    def classify(file_path):  ##TO classify the image
        image = Image.open(file_path)
        newSize = (32, 32)
        image.resize(newSize)
        img = cv.imread(file_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        plt.imshow(img, cmap=plt.cm.binary)
        prediction = model.predict(np.array([img]) / 255)
        index = np.argmax(prediction)
        print(f'Prediction is {class_name[index]}')
        label.configure(foreground="#011683", text="The Prediction is" + class_name[index])

    # initialize Second GUI

    front = tk.Tk()

    # set the heightxwidth

    front.geometry('800x600')
    front.title("Image Classification Project")
    front.config(background="#CDCDCD")

    # set Heading

    heading = Label(front, text="Image Classification", pady=20, font=('arial', 20, 'bold'))

    heading.configure(background="#CDCDCD", foreground="#364156")

    heading.pack()  # Pack the heading with front objects

    upload = Button(front, text="Upload Image", command=Upload_Image, padx=10,
                    pady=5)  # Command defines what function needs to be performed when this button is pressed
    upload.configure(background="#364156", foreground="white", font=('arial', 20, 'bold'))
    upload.pack(side=BOTTOM, pady=50)

    sign_image = Label(front)  # Here will show the uploaded Image
    sign_image.pack(side=BOTTOM, expand=True)

    # Predicted Image
    label = Label(front, background="#CDCDCD", font=('arial', 20, 'bold'))
    label.pack(side=BOTTOM, expand=True)

    front.mainloop()


# creates a Tk() object
master = Tk()

# sets the geometry of main
# root window
master.geometry("600x600")
master.title("Image Classification Project")
master.config(background="#CDCDCD")

c=Canvas(master,bg="gray16",height=320,width=600)

image = ImageTk.PhotoImage(Image.open("C:\\Users\\talha\OneDrive\\Desktop\\background.jpeg"))

c.create_image(0, 0, anchor="nw", image=image)
c.pack()

heading = Label(master, text="Image Classification Project", pady=20, font=('arial', 20, 'bold'), background="#CDCDCD")
heading.place(relx=0.20, rely=0.61)

btn = Button(master, text="Get Started", command=newWindow, height=2, width=15, font=('arial', 14, 'bold'))
btn.config(foreground="#FFFFFF", background="#42adf5")
btn.pack(side=BOTTOM, pady=20, padx=15)

# mainloop, runs infinitely
master.mainloop()