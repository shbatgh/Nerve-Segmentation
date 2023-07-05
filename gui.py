import customtkinter as ctk
from customtkinter import CTkImage
from tkinter import filedialog, Label
from PIL import Image, ImageTk
from fastai.vision import load_learner, open_image
import numpy as np
from fastai.vision import pil2tensor, open_image
import numpy as np
import PIL
from PIL import Image 
import os

# Load the model from 
model = load_learner('C:/Users/grand/dev/internship2023', 'Unet_processing')

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Segmentation")
        self.geometry("600x320")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.model = model

        self.input_button = ctk.CTkButton(self, text="Input Image", command=self.open_image)
        self.input_button.grid(row=1, column=0, padx=10, pady=0, sticky="ns")

        self.segment_button = ctk.CTkButton(self, text="Segment", command=self.segment_image)
        self.segment_button.grid(row=1, column=1, padx=10, pady=0, sticky="ns")

        self.output_button = ctk.CTkButton(self, text="Output Image", command=self.button_callback)
        self.output_button.grid(row=1, column=2, padx=10, pady=0, sticky="ns")

        self.download_button = ctk.CTkButton(self, text="Download Image", command=self.download_image)
        self.download_button.grid(row=0, column=2, padx=10, pady=20)

        self.download_button = ctk.CTkButton(self, text="Load Model", command=self.load_model)
        self.download_button.grid(row=0, column=1, padx=0, pady=0)

        self.input_image_label = ctk.CTkLabel(self, text="", fg_color="transparent")
        self.input_image_label.grid(row=2, column=0, padx=10, pady=10)

        self.output_image_label = ctk.CTkLabel(self, text="", fg_color="transparent")
        self.output_image_label.grid(row=2, column=2, padx=10, pady=10)

        self.rowconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        self.rowconfigure(1, weight=0)

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)


    def open_image(self):
        file_path = filedialog.askopenfilename()
        self.image_path = file_path

        image = Image.open(file_path)
        image = image.resize((256, 256))
        self.image = open_image(file_path)
        photo = ImageTk.PhotoImage(image)

        self.input_image_label.configure(image=photo)
        self.input_image_label.image = photo

        return image

    def segment_image(self):
        image = self.inference(self.image_path)
        self.output_image = image

        photo = ImageTk.PhotoImage(image)

        self.output_image_label.configure(image=photo)
        self.output_image_label.image = photo

        return photo
    
    def download_image(self):
        file_path = filedialog.asksaveasfilename()
        self.output_image.save(file_path)

    def load_model(self):
        model_path = filedialog.askopenfilename()
        self.model = load_learner(os.path.dirname(model_path), os.path.basename(model_path))

    def inference(self, image_path):
        minv_global = 1000;
        maxv_global = -1000;
        with Image.open(image_path) as im:
            test = open_image(image_path)
            inf = self.model.predict(open_image(image_path))
            #Convert from Fast ai to numpy and convert the dimension
            inf_numpy = inf[2].numpy()
            inf_numpy = inf_numpy[0,:,:]
            minv = np.amin(inf_numpy)
            maxv = np.amax(inf_numpy)
            if minv < minv_global:
                minv_global = minv
            if maxv > maxv_global:
                maxv_global = maxv
            #Convert from numpy to Pillow
            inf_numpy = (255 * (inf_numpy - minv_global) / (maxv_global - minv_global)).astype(np.uint8)
            im2 = PIL.Image.fromarray(inf_numpy)

        return im2

    def button_callback(self):
        print("button pressed")

app = App()
app.mainloop()
