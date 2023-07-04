import customtkinter as ctk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
from fastai.vision import load_learner, open_image
import numpy as np
from fastai.vision import pil2tensor, open_image
import numpy as np
import PIL
from PIL import Image 
import torch

# Load the model from 
model = load_learner('C:/Users/grand/dev/internship2023/TRAINING/ForTraining3/FOR ARTICLE/INFERENCE/NEW_MODELS/34_1/U_net', 'U-net-final')

def inference(image_path):
    minv_global = 1000;
    maxv_global = -1000;
    with Image.open(image_path) as im:
        pic = pil2tensor(im, np.float32)
        pic2 = torch.cat((pic, pic,pic), dim=0)
        pic3 = torch.div(pic, 255)
        inf = model.predict(open_image(image_path))
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

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Segmentation")
        self.geometry("600x300")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.input_button = ctk.CTkButton(self, text="Input Image", command=self.open_image)
        self.input_button.grid(row=1, column=0, padx=10, pady=0, sticky="ns")

        self.segment_button = ctk.CTkButton(self, text="Segment", command=self.segment_image)
        self.segment_button.grid(row=1, column=1, padx=10, pady=0, sticky="ns")

        self.output_button = ctk.CTkButton(self, text="Output Image", command=self.button_callback)
        self.output_button.grid(row=1, column=2, padx=10, pady=0, sticky="ns")

        self.download_button = ctk.CTkButton(self, text="Download Image", command=self.download_image)
        self.download_button.grid(row=3, column=2, padx=10, pady=20, sticky="ns")

        self.image_label = ctk.CTkLabel(self, text="", fg_color="transparent")
        self.image_label.grid(row=2, column=0, padx=10, pady=10)

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
        photo = ctk.PhotoImage(image)

        self.image_label = ctk.CTkLabel(image=photo)
        self.image_label.image = photo

        return image

    def segment_image(self):
        image = inference(self.image_path)
        self.output_image = image

        photo = ImageTk.PhotoImage(image)

        self.output_image_label.configure(image=photo)
        self.output_image_label.image = photo

        return photo
    
    def download_image(self):
        file_path = filedialog.asksaveasfilename()
        self.output_image.save(file_path)

    def button_callback(self):
        print("button pressed")

app = App()
app.mainloop()