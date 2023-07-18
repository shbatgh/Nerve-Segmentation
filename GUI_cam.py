# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 13:54:23 2023

@author: salhadda
"""

import tkinter
import tkinter.messagebox
import customtkinter
import cv2
from PIL import Image, ImageTk
import sys
# from customtkinter import messagebox
import time
import threading
import numpy as np
from tifffile import imwrite


from collections import deque##### for te rolling buffer of the saved images





# ##### initialization of the cameras:
# cap1 = cv2.VideoCapture(1)
# cap0 = cv2.VideoCapture(0)


##### end initialization of the cameras
customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        
        # create the close button
        self.protocol("WM_DELETE_WINDOW", self.on_closing) 
        
        
        
        # ########### initialization for the arduino port
        
        # self.port="COM3"
        # self.board=pyfirmata.Arduino(self.port)
        # print('communication with arduino establised successfully!')
        # ############ end initialization arduino port
        
        
        ####### initialization part of the cameras
        
        # for the main camera
        self.video_source=0#### modified for now should always 1
        self.vid=MyVideoCapture(self.video_source)
        # for the macro camera
        # self.video_sourceM=2
        # self.vidM=MyVideoCapture(self.video_sourceM)
        
        ####### end initialization part of the cameras


############
        # Initialize deque with maxlen 100
        self.image_buffer = deque(maxlen=100)
############



        # configure window
        self.title("test.py")
        self.geometry(f"{1300}x{780}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=50, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)


###### add a section to show the saturation of the camera and the frame rate
                ########### part for the saturation of the camera
        self.sidebar_text_11 = customtkinter.CTkLabel(self.sidebar_frame, text="Saturation(255):", font=customtkinter.CTkFont(size=14, weight="bold"))
        self.sidebar_text_11.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")
        self.sidebar_label_21 = customtkinter.CTkLabel(self.sidebar_frame, text="")
        self.sidebar_label_21.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

                ########### part for the display rate
        self.sidebar_text_12 = customtkinter.CTkLabel(self.sidebar_frame, text="Display rate (fps):", font=customtkinter.CTkFont(size=14, weight="bold"))
        self.sidebar_text_12.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        self.sidebar_label_22 = customtkinter.CTkLabel(self.sidebar_frame, text="")
        self.sidebar_label_22.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")

###### end part for the saturation of the camera and the frame rate

        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Parameters", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=4, column=0, padx=20, pady=(20, 10))



        self.sidebar_text_1 = customtkinter.CTkLabel(self.sidebar_frame, text="ID #:")
        self.sidebar_text_1.grid(row=5, column=0, padx=20, pady=10, sticky="nsew")
        self.sidebar_entry_2 = customtkinter.CTkEntry(self.sidebar_frame, placeholder_text="Patient ID")
        self.sidebar_entry_2.grid(row=6, column=0, padx=20, pady=10, sticky="nsew")
        ### part for the LED power
        self.sidebar_text_2 = customtkinter.CTkLabel(self.sidebar_frame, text="LED power:")
        self.sidebar_text_2.grid(row=7, column=0, padx=20, pady=10, sticky="nsew")
        self.sidebar_entry_3 = customtkinter.CTkEntry(self.sidebar_frame, placeholder_text="1000")
        self.sidebar_entry_3.grid(row=8, column=0, padx=20, pady=10, sticky="w")
        ### end part of the LED power
        
        ### to add notes
        # self.sidebar_text_3 = customtkinter.CTkLabel(self.sidebar_frame, text="Remarks")
        # self.sidebar_text_3.grid(row=5, column=0, padx=20, pady=10, sticky="nsew")
        # self.sidebar_entry_4 = customtkinter.CTkEntry(self.sidebar_frame, placeholder_text="notes to be added")
        # self.sidebar_entry_4.grid(row=6, column=0, padx=20, pady=10, sticky="nsew")
        ### end of notes to be added
        
        
        self.radio_var = tkinter.IntVar(value=0)
        self.radio_button_1 = customtkinter.CTkRadioButton(self.sidebar_frame, variable=self.radio_var, value=0,text="averaging")
        self.radio_button_1.grid(row=9, column=0, pady=10, padx=20, sticky="nsew")
        self.radio_button_2 = customtkinter.CTkRadioButton(self.sidebar_frame, variable=self.radio_var, value=1,text="volume imaging")
        self.radio_button_2.grid(row=10, column=0, pady=10, padx=20, sticky="nsew")
        
        ######## part to add a note text
        
        self.sidebar_entry_4 = customtkinter.CTkEntry(self.sidebar_frame, placeholder_text="notes to be added")
        self.sidebar_entry_4.grid(row=11, column=0, padx=20, pady=10, sticky="nsew")
        ####### end part note text
        
        # self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, command=self.sidebar_button_event)
        # self.sidebar_button_3.grid(row=5, column=0, padx=20, pady=10)
        
        
        # self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="GUI parameters", font=customtkinter.CTkFont(size=20, weight="bold"))
        # self.logo_label.grid(row=9, column=0, padx=20, pady=(20, 10))
        
        
        
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="GUI appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=12, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=13, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=14, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=15, column=0, padx=20, pady=(10, 20))






        # create main entry and button
        # self.entry = customtkinter.CTkEntry(self, placeholder_text="to add notes")
        # self.entry.grid(row=3, column=1, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

        self.main_button_1 = customtkinter.CTkButton(master=self, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), text="acquire",command=self.acquisition)
        self.main_button_1.grid(row=3, column=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # create textbox
        # self.textbox = customtkinter.CTkTextbox(self, width=250)
        # self.textbox.grid(row=0, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")

## part to  create a frame to display live images


        self.mainframe_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.mainframe_frame.grid(row=0, column=1, rowspan=7, columnspan=2, padx=10, pady=10, ipadx=5, ipady=5, sticky="nsew")
        self.mainframe_frame.grid_rowconfigure(0, weight=1)
        
        ### create label inside the created frame
        
        # self.mainframe_label=customtkinter.CTkLabel(self.mainframe_frame)
        # self.mainframe_label.grid(row=0, column=0, sticky="nsew")
        
        # self.mainframe_label.grid_rowconfigure(4, weight=1)
        
        
        
        self.mainframe_label =customtkinter.CTkLabel(self.mainframe_frame, text="", compound=customtkinter.CENTER, anchor=customtkinter.CENTER)
        # lmain1.pack()
        # self.mainframe_label.pack(side='top', anchor='center')
        self.mainframe_label.grid(column=0, row=0)

        
### use the grid approach
        self.webcam_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.webcam_frame.grid(row=0, column=3, padx=(5, 5), pady=(5, 0), sticky="nsew")
        self.webcam_frame.grid_rowconfigure(0, weight=1)
        
       
        self.webcam_label =customtkinter.CTkLabel(self.webcam_frame, text="", compound=customtkinter.CENTER, anchor=customtkinter.CENTER)
        # self.webcam_label.pack(side='top', anchor='center')
        self.webcam_label.grid(column=0, row=0)

        
        #################### zoomed images frame: I am trying to use the harvesters library in this case to have the best resolution at a high speed
        
        self.zoom_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.zoom_frame.grid(row=1, column=3, padx=(5, 5), pady=(5, 0), sticky="nsew")
        self.zoom_frame.grid_rowconfigure(0, weight=1)
        
        self.zoom_label =customtkinter.CTkLabel(self.zoom_frame, text="", compound=customtkinter.CENTER, anchor=customtkinter.CENTER)
        # self.zoom_label.pack(side='top', anchor='center')
        self.zoom_label.grid(column=0, row=0)

        ################# part for the sliders

        ## create slider and progressbar frame
        self.slider_progressbar_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.slider_progressbar_frame.grid(row=3, column=1, columnspan=1, padx=(5, 0), pady=(5, 0), sticky="nsew")
        self.slider_progressbar_frame.grid_columnconfigure(0, weight=1)
        self.slider_progressbar_frame.grid_rowconfigure(4, weight=1)
        
        # self.seg_button_1 = customtkinter.CTkSegmentedButton(self.slider_progressbar_frame)
        # self.seg_button_1.grid(row=0, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")
        # self.progressbar_1 = customtkinter.CTkProgressBar(self.slider_progressbar_frame)
        # self.progressbar_1.grid(row=1, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")
        self.progressbar_2 = customtkinter.CTkProgressBar(self.slider_progressbar_frame)
        self.progressbar_2.grid(row=2, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")
        self.slider_1 = customtkinter.CTkSlider(self.slider_progressbar_frame, from_=0, to=1, number_of_steps=200)
        self.slider_1.grid(row=3, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")
        #### in order to find the value of the slider to change the tunnable lens voltage
        ## we can use the 
        # self.slider_2 = customtkinter.CTkSlider(self.slider_progressbar_frame, orientation="vertical")
        # self.slider_2.grid(row=0, column=1, rowspan=5, padx=(10, 10), pady=(10, 10), sticky="ns")
        # self.progressbar_3 = customtkinter.CTkProgressBar(self.slider_progressbar_frame, orientation="vertical")
        # self.progressbar_3.grid(row=0, column=2, rowspan=5, padx=(10, 20), pady=(10, 10), sticky="ns")

        
        ################# end part for the sliders
        
        
        # # set default values
        # # self.sidebar_button_3.configure(state="disabled", text="Disabled CTkButton")
        # self.checkbox_2.configure(state="disabled")
        # self.switch_2.configure(state="disabled")
        # self.checkbox_1.select()
        # self.switch_1.select()
        # # self.radio_button_3.configure(state="disabled")
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")
        # self.optionmenu_1.set("CTkOptionmenu")
        # self.combobox_1.set("CTkComboBox")
        
        
        ##### part for the progress bar and sliders
        
        
        self.slider_1.configure(command=self.progressbar_2.set)

        
        ############################################################ call trigger in the background
        ########################### run the background loop
        # t = threading.Thread(target=self.Triggering)
        # t.start()
        ########################### end the background loop
        

    def open_input_dialog_event(self):
        dialog = customtkinter.CTkInputDialog(text="Type in a number:", title="CTkInputDialog")
        print("CTkInputDialog:", dialog.get_input())

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def sidebar_button_event(self):
        print("sidebar_button click")
        
    def on_closing(self):
        print("close command")
        self.destroy()
        sys.exit
        
    def acquisition(self):
        print("acquisition in process")
        

        
        # id_number = self.sidebar_entry_2.get()
        # if not id_number:
        #     print("missing ID number")
        #     # self.sidebar_entry_2.set("Fill ID#")
        # else:        
        # Create a 3D numpy array from deque
        self.image_stack = np.stack(self.image_buffer)
        
        # Save the image stack as a TIFF file
        
        
        current_time = time.strftime("__%Y-%m-%d_%H-%M-%S", time.gmtime())
        # filename="C:/Users/Slava/Desktop/testSaving/"f"{current_time}"+".tif"
        filename="testImages/"f"{current_time}"+".tif"

        imwrite(filename, self.image_stack)

    
    def update(self):
        # print('in')
        start_time = time.time()
        
        ret, frame = self.vid.get_frame()
        # retM, frameM = self.vidM.get_frame()
        # print(frame.shape)
        
        if ret:
            # self.photo=ImageTk.PhotoImage(image=Image.fromarray(frame))
            
            
            ##########################################    add a part for gaussian blurring
            
            # frame_blur = cv2.GaussianBlur(frame, (3,3), 0)
            # # print(np.mean(frame))

            
            
            # ##########################################   end part of gaussian blurringx
            x=self.mainframe_frame.winfo_width()
            y=self.mainframe_frame.winfo_height()
            # self.photo=customtkinter.CTkImage(Image.fromarray(frame-frame_blur), size=(x, y))#### modified for ctkinter
            self.photo=customtkinter.CTkImage(Image.fromarray(frame), size=(x, y))#### modified for ctkinter
            self.image_buffer.append(frame)### append the rolling buffer

################## display saturation
########## to display the saturation level

            self.sidebar_label_21.configure(text=str(float((np.mean(frame)))))

            
######### end part for the saturation level

            # self.mainframe_label.imgtk = self.photo
            # self.mainframe_label.configure(image= self.photo, width=600, height=600)## initial size of the label
            self.mainframe_label.configure(image= self.photo, width=x, height=y)

########## to display the acquisition frame rate
            elapsed_time = time.time() - start_time
            fps = 1 / elapsed_time
            self.sidebar_label_22.configure(text=str(float(fps)))

            
######### end part for the acquisition frame rate

            # self.mainframe_label.pack()
            self.mainframe_label.grid(column=0, row=0)

            
            # self.webcam_label.configure(image= self.photoM, width=400, height=400)
            # # self.webcam_label.pack()
            # self.webcam_label.grid(column=0, row=0)
            
            # # self.mainframe_frame.grid(row=0, column=1, rowspan=8, columnspan=2, sticky="nsew")
            # # self.mainframe_frame.grid_rowconfigure(4, weight=1)
            
            # # self.mainframe_label.pack(side="top",expand="yes", fill="both")
            
            # ### for the zoomed display
            # self.zoom_label.configure(image= self.photoM, width=400, height=400)
            # self.zoom_label.grid(column=0, row=0)
            
        self.mainframe_frame.after(10,self.update)### run the function again after a certain delay
        
        
    def Triggering(self):
    ########################## part for the LED triggerinf illumination        
        while(True):
            end_time = time.perf_counter() + 0.01
            while time.perf_counter() < (end_time):####time of the on pulse
                # print(end_time)
                self.board.digital[2].write(1)
                # print(self.board.digital[2].write(1))
            while time.perf_counter() < (end_time+ 0.04):####second cycle including the on pulse
            #     # print('hello world2')
                self.board.digital[2].write(0)
               
                
                
                ######################################################
            # print("Background loop running")
            time.sleep(0.05)
            ##########################################################3
            # print('threaded')
            # pass
                        
        
        ################end part LED illumination        pass
        

        
        
        
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
                
            
            

        
        
        
        # #################################################### part to know available properties of the camera in use
        # print("Available camera properties:")
        # for i in range(0, 100):
        #     prop = self.vid.get(i)
        #     if prop is not None:
        #         print(i, ": ", prop)
                
                
                
                
        #                 ###############################
        #         # define dictionary of property IDs and names
        # prop_dict = {
        #     cv2.CAP_PROP_POS_MSEC: 'CAP_PROP_POS_MSEC',
        #     cv2.CAP_PROP_POS_FRAMES: 'CAP_PROP_POS_FRAMES',
        #     cv2.CAP_PROP_POS_AVI_RATIO: 'CAP_PROP_POS_AVI_RATIO',
        #     cv2.CAP_PROP_FRAME_WIDTH: 'CAP_PROP_FRAME_WIDTH',
        #     cv2.CAP_PROP_FRAME_HEIGHT: 'CAP_PROP_FRAME_HEIGHT',
        #     cv2.CAP_PROP_FPS: 'CAP_PROP_FPS',
        #     cv2.CAP_PROP_FOURCC: 'CAP_PROP_FOURCC',
        #     cv2.CAP_PROP_BRIGHTNESS: 'CAP_PROP_BRIGHTNESS',
        #     cv2.CAP_PROP_CONTRAST: 'CAP_PROP_CONTRAST',
        #     cv2.CAP_PROP_SATURATION: 'CAP_PROP_SATURATION',
        #     cv2.CAP_PROP_HUE: 'CAP_PROP_HUE',
        #     cv2.CAP_PROP_EXPOSURE: 'CAP_PROP_EXPOSURE',
        #     cv2.CAP_PROP_GAIN: 'CAP_PROP_GAIN',
        #     cv2.CAP_PROP_AUTO_EXPOSURE: 'CAP_PROP_AUTO_EXPOSURE',
        #     cv2.CAP_PROP_AUTOFOCUS: 'CAP_PROP_AUTOFOCUS',
        #     cv2.CAP_PROP_ZOOM: 'CAP_PROP_ZOOM'
        # }
        
        # # print the names of the first 100 camera properties
        # print("First 100 camera properties:")
        # for i in range(0, 100):
        #     if i in prop_dict:
        #         prop_name = prop_dict[i]
        #         prop_value = self.vid.get(i)
        #         print(i, ": ", prop_name, "-", prop_value)

        # ################################################### end part to know available parameters to change
        
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
        
        

            
        
        
        
        
        
        
        ########### end part for frame update


if __name__ == "__main__":
    
    app = App()

    # app.showframe()
    app.update()#### added for the updates of the frame of the live display
    ########################### run the background loop
    # t = threading.Thread(target=app.Triggering)
    # t.start()
    ########################### end the background loop
    app.mainloop()
    

    