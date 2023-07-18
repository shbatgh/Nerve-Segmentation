import tkinter
import tkinter.messagebox
import customtkinter
import cv2
import PIL
import time
import fastai
from fastai.vision import load_learner, pil2tensor, Image
import numpy as np
import PIL
import torch
from collections import deque
from tifffile import imwrite

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


model = load_learner('../internship2023', 'Unet_processing')

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.model = model
        self.image_buffer = deque(maxlen=100)

        ####### initialization part of the cameras
        self.video_source=0
        self.vid = MyVideoCapture(self.video_source)

        # configure window
        self.title("Live Segmentation")
        self.geometry(f"{1200}x{700}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=50, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)


        ###### add a section to show the frame rate
        self.sidebar_text_12 = customtkinter.CTkLabel(self.sidebar_frame, text="Display rate (fps):", font=customtkinter.CTkFont(size=14, weight="bold"))
        self.sidebar_text_12.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        self.sidebar_label_22 = customtkinter.CTkLabel(self.sidebar_frame, text="")
        self.sidebar_label_22.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")


        ## part to  create a frame to display live images
        self.mainframe_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.mainframe_frame.grid(row=0, column=1, rowspan=7, columnspan=2, padx=10, pady=10, ipadx=5, ipady=5, sticky="nsew")
        self.mainframe_frame.grid_rowconfigure(0, weight=1)

        self.mainframe_label = customtkinter.CTkLabel(self.mainframe_frame, text="")
        #self.mainframe_label.grid(column=0, row=0)

        self.nmainframe_label = customtkinter.CTkLabel(self.mainframe_frame, text="")
        #self.nmainframe_label.grid(column=1, row=0)

        self.acquire_button = customtkinter.CTkButton(master=self, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), text="acquire",command=self.acquisition)
        self.acquire_button.grid(row=3, column=3, padx=(20, 20), pady=(20, 20), sticky="nsew")


    def inference(self, img):
        minv_global = 1000;
        maxv_global = -1000;

        inf = self.model.predict(img)
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
    
    def acquisition(self):
        print("acquisition in process")
        
        self.image_stack = np.stack(self.image_buffer)
        
        current_time = time.strftime("__%Y-%m-%d_%H-%M-%S", time.gmtime())
        filename="testImages/"f"{current_time}"+".tif"

        imwrite(filename, self.image_stack)

    def update(self):
        start_time = time.time()
        
        ret, frame = self.vid.get_frame()
        if ret:
            x = self.mainframe_frame._current_width/2
            y = self.mainframe_frame._current_height/2
            #self.photo = customtkinter.CTkImage(PIL.Image.fromarray(frame), size=(x, y))#### modified for ctkinter
            #PIL.Image.fromarray(frame).save("C:/Users/grand/dev/internship2023/1.jpg")
            # convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # downsize gray to half its original size
            gray = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25)
            self.photo = self.inference(Image(pil2tensor(gray, np.float32).div_(255)))
            self.image_buffer.append(np.array(self.photo))
            self.photo = customtkinter.CTkImage(self.photo, size=(x, y))

            frame = customtkinter.CTkImage(PIL.Image.fromarray(frame), size=(x, y))

            self.mainframe_label.configure(image = frame, width=x, height=y)
            self.nmainframe_label.configure(image = self.photo, width=x, height=y)

            ########## to display the acquisition frame rate
            elapsed_time = time.time() - start_time
            fps = 1 / elapsed_time
            self.sidebar_label_22.configure(text=str(float(fps)))
  
            self.mainframe_label.grid(column=0, row=0)
            self.nmainframe_label.grid(column=1, row=0)


            
        self.mainframe_frame.after(10,self.update)### run the function again after a certain delay
     
class MyVideoCapture:
    
    def __init__(self, video_source=0):
        self.vid=cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise Exception("Could not open the video source", video_source)
            
            
            ########################### for main camera, imaging camera
        if video_source==1:
            # ########### choose the frame rate
            self.vid.set(cv2.CAP_PROP_FPS, 10)
            # # set exposure time to 98 milliseconds
            self.vid.set(cv2.CAP_PROP_EXPOSURE, 98000)
            # # set the gain to 100
            # self.vid.set(cv2.CAP_PROP_GAIN, 100)
            self.vid.set(cv2.CAP_PROP_GAIN, 10)
            
            
            
            
            ################## set the resolution of the camera in use
            self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 2000)
            self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 2000)
            
            print( str(int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            
            print("the size of the frame is printed")
            
            ################## end set the resolution camers in use
            
            
            
            ################## for secondary macro camera
            
        if video_source==2:
            # ########### choose the frame rate
            self.vid.set(cv2.CAP_PROP_FPS, 10)
            # # set exposure time to 98 milliseconds
            self.vid.set(cv2.CAP_PROP_EXPOSURE, 98000)
            # # set the gain to 100
            # self.vid.set(cv2.CAP_PROP_GAIN, 100)
            self.vid.set(cv2.CAP_PROP_GAIN, 10)
                
        # ############ get the exposure time and the frame rate
        
        print(self.vid.get(cv2.CAP_PROP_FPS))# for frame rate
        print(self.vid.get(cv2.CAP_PROP_EXPOSURE))# for exposure time
        
        ###########
            # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
            # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)



if __name__ == "__main__":
    
    app = App()

    # app.showframe()
    app.update()#### added for the updates of the frame of the live display
    ########################### run the background loop
    # t = threading.Thread(target=app.Triggering)
    # t.start()
    ########################### end the background loop
    app.mainloop()
    

    