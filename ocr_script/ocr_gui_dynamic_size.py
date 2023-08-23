# For dynamic window and widget resizing (needs to come before tkinter import)
import ctypes
ctypes.windll.shcore.SetProcessDpiAwareness(1)

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
import screeninfo
import numpy as np
import cv2
from cv2 import aruco
import gxipy as gx
from PIL import Image, ImageTk
from ocr_code import ocr_on_roi, crop_roi
import roi_calib as rc
import img_processing
import ocr_code
import database
import pickle
import datetime
import time
from sympy import symbols, Eq
import threading
import copy
import re
import os


class Rectangle:

    def __init__(self, canvas, roi_list_container, rect_number, coordinates = (), variable = "", only_numbers = True, detect_id = 0, font_id = 0):
        
        self.canvas = canvas
        self.roi_list_container = roi_list_container
        self.rect_number = rect_number
        self.coordinates = coordinates
        self.isactive = True
        
        # Rectangle drawing parameters
        self.drawing_label = ttk.Label(self.canvas, text=str(rect_number), font=('Arial', 12), background='white', foreground='black')
        drawing_label_x = (self.coordinates[0] + self.coordinates[2]) // 2
        drawing_label_y = self.coordinates[1] - 15
        self.drawing_label.place(x=drawing_label_x, y=drawing_label_y, anchor='center')
        self.rectangle_drawing = self.canvas.create_rectangle(self.coordinates, outline='red', width=1)

        # ROI number label
        self.var_label = ttk.Label(self.roi_list_container, text=str(rect_number), font=('calibre',10, 'bold'))
        self.var_label.grid(row=rect_number+1,column=0)

        # Variable name entry
        self.variable = tk.StringVar(value=variable)
        self.rect_entry = ttk.Entry(self.roi_list_container,textvariable = self.variable, font=('calibre',10,'normal'))
        self.rect_entry.grid(row=rect_number+1,column=1)

        # Only numbers checkbox
        self.only_numbers = tk.BooleanVar(value=only_numbers)
        self.only_num_checkbox = ttk.Checkbutton(self.roi_list_container, variable=self.only_numbers)
        self.only_num_checkbox.grid(row=rect_number+1,column=2)

        # Detection method dropdown
        self.detection_dropdown = ttk.Combobox(self.roi_list_container, value=["Fastest", "Best"], width= 7)
        self.detection_dropdown.current(detect_id)
        self.detection_dropdown.grid(row=rect_number+1,column=3)

        # Font dropdown
        self.font_type_dropdown = ttk.Combobox(self.roi_list_container, value=[font.name for font in img_processing.fonts], width=12)
        self.font_type_dropdown.current(font_id)
        self.font_type_dropdown.grid(row=rect_number+1,column=4)

        # Delete button
        self.delete_btn = ttk.Button(self.roi_list_container,text = 'Delete', command=lambda: self.delete_rect())
        self.delete_btn.grid(row=rect_number+1,column=5)

    def delete_rect(self):
        '''Function deleting the components related to this Rectangle object'''

        self.isactive = False
        self.drawing_label.destroy()
        self.canvas.delete(self.rectangle_drawing)
        self.var_label.destroy()
        self.rect_entry.destroy()
        self.only_num_checkbox.destroy()
        self.detection_dropdown.destroy()
        self.font_type_dropdown.destroy()
        self.delete_btn.destroy()



class OCR_GUI:

    def __init__(self, master):

        # Main window
        self.master = master
        self.master.title("OCR GUI")

        # Get screen width and height
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()

        # Adjust sizes based on screen dimensions
        self.right_frame_width = 650
        self.left_frame_parameter_height = 240
        self.canvas_max_width = screen_width - self.right_frame_width
        self.canvas_max_height = screen_height - self.left_frame_parameter_height

        # Left frame
        self.left_frame = tk.Frame(master, width=self.canvas_max_width)
        self.left_frame.pack(side=tk.LEFT, fill='both', padx=15, pady=10)

        # Right frame
        self.right_frame = tk.Frame(master, width=self.right_frame_width)
        self.right_frame.pack(side=tk.RIGHT, fill='both', padx=15, pady=15)

        # Bind the configure event for window size adjustments
        self.master.bind("<Configure>", self.adjust_frame_sizes)

        ############################### Left frame ###############################
        
        # Creating radiobuttons to choose camera type
        self.general_params_frame = tk.Frame(self.left_frame, highlightbackground = "grey", highlightthickness = 1)
        self.general_params_frame.pack(side=tk.TOP)

        # Camera type radio buttons
        self.cam_type = tk.StringVar(value="daheng")
        self.cam_type_label = ttk.Label(self.general_params_frame, text="Select type of camera:")
        self.daheng_radio_btn = ttk.Radiobutton(self.general_params_frame, text="Daheng camera", variable=self.cam_type, value="daheng", command=self.update_cam_type)
        self.webcam_radio_btn = ttk.Radiobutton(self.general_params_frame, text="Webcam", variable=self.cam_type, value="webcam", command=self.update_cam_type)
        self.cam_type_label.grid(row=0, column=0, padx=30, pady=(10,5))
        self.daheng_radio_btn.grid(row=0, column=1, padx=5, pady=(10,5))
        self.webcam_radio_btn.grid(row=0, column=2, padx=(5,30), pady=(10,5))
        self.device_manager = gx.DeviceManager()
        self.selected_cam_type = self.cam_type.get()

        # Daheng camera settings frame
        self.daheng_settings_frame = tk.Frame(self.general_params_frame)
        self.daheng_settings_frame.grid(row=1, column=0, columnspan = 3)

        # Entry widget for exposure time
        self.exposure_time_label = ttk.Label(self.daheng_settings_frame, text="Exposure Time [ms]:")
        self.exposure_time_label.grid(row=0, column=0, padx=5, pady=(5, 5))
        self.exposure_time_entry = ttk.Entry(self.daheng_settings_frame)
        self.exposure_time_entry.grid(row=0, column=1, padx=5, pady=(5, 5))
        self.exposure = 100
        self.exposure_time_entry.insert(-1, str(self.exposure))
        self.exposure_time_entry.bind("<Return>", lambda event: on_exposure_entry())

        def on_exposure_entry():
            try:    
                exposure_time = int(self.exposure_time_entry.get())
                if 1 <= exposure_time <= 999:
                    self.exposure = exposure_time
                    self.cam.ExposureTime.set(self.exposure * 1000)
                else:
                    messagebox.showerror("Error", "Exposure time must be between 1 and 999")
            except ValueError:
                messagebox.showerror("Error", "Exposure time must be an integer")
        
        # Entry widget for analog gain
        self.gain_label = ttk.Label(self.daheng_settings_frame, text="Analog gain:")
        self.gain_label.grid(row=0, column=2, padx=5, pady=(5, 5))
        self.gain_entry = ttk.Entry(self.daheng_settings_frame)
        self.gain_entry.grid(row=0, column=3, padx=5, pady=(5, 5))
        self.gain = 10
        self.gain_entry.insert(-1, str(self.gain))
        self.gain_entry.bind("<Return>", lambda event: on_gain_entry())

        def on_gain_entry():
            try:    
                gain = int(self.gain_entry.get())
                self.gain = gain
                self.cam.Gain.set(self.gain)
                print("Gain changed to ", self.gain)
            except ValueError:
                messagebox.showerror("Error", "Gain must be an integer")

        # Button for auto balance white
        self.autobalancewhite_button = ttk.Button(self.daheng_settings_frame, text="Auto Balance White", command=lambda: self.cam.BalanceWhiteAuto.set(1))
        self.autobalancewhite_button.grid(row=0, column=4, padx=15, pady=(5, 5))

        # Image canvas for corrected view of target surface
        self.canvas = tk.Canvas(self.left_frame, bd=2, bg="grey", width=self.canvas_max_width, height=self.canvas_max_height)
        self.canvas.pack(padx=15, pady=(10,5))
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW)
        self.last_rvec = None
        self.last_tvec = None
        self.surface = None

        # Create a container frame for the buttons
        button_container = ttk.Frame(self.left_frame)
        button_container.pack(side=tk.BOTTOM, pady=(5, 10))

        # "Indicate display surface" button
        self.indicate_surface_button = ttk.Button(button_container, text="Indicate display surface", command=self.indicate_surface_window_init)
        self.indicate_surface_button.pack(side=tk.LEFT, padx=15)

        # "Export settings" button
        self.export_button = ttk.Button(button_container, text="Export settings", command=self.export_parameters)
        self.export_button.pack(side=tk.LEFT, padx=15)

        # "Import settings" button 
        self.import_button = ttk.Button(button_container, text="Import settings", command=self.import_parameters)
        self.import_button.pack(side=tk.LEFT, padx=15)


        ############################### Right frame ###############################
        
        # Create new frame to hold the parameters grid
        self.parameters_frame = tk.Frame(self.right_frame, highlightbackground = "grey", highlightthickness = 1)
        self.parameters_frame.pack(fill=tk.X, padx=5, pady=15)

        # Camera input choice dropdown menu
        self.cam_input = ttk.Label(self.parameters_frame, text="Camera:")
        self.cam_input.grid(row=0, column=0, padx=5, pady=(20, 5))
        self.camera_inputs = self.get_available_cameras()
        self.camera_input_dropdown = ttk.Combobox(self.parameters_frame, value=self.camera_inputs)
        self.camera_input_dropdown.current(0)
        self.selected_camera_input = self.camera_input_dropdown.get()
        self.camera_input_dropdown.grid(row=0, column=1, padx=5, pady=(20, 5))
        self.camera_input_dropdown.bind("<<ComboboxSelected>>", lambda event: self.update_cam_input())
        self.import_button = ttk.Button(self.parameters_frame, text="Refresh", command=self.refresh_cam_inputs)
        self.import_button.grid(row=0, column=2, padx=5, pady=(20, 5))

        # Camera calibration dropdown menu
        self.cam = None
        self.cam_calib_label = ttk.Label(self.parameters_frame, text="Calibration folder:")
        self.cam_calib_label.grid(row=1, column=0, padx=5, pady=5)
        self.calibration_names = [entry for entry in os.listdir("cam_calibration/cameras") if os.path.isdir(os.path.join("cam_calibration/cameras", entry))]
        self.calibration_name_dropdown = ttk.Combobox(self.parameters_frame, value=self.calibration_names)
        self.calibration_name_dropdown.current(0)
        self.calibration_name_dropdown.grid(row=1, column=1, padx=5, pady=5)
        self.calibration_name_dropdown.bind("<<ComboboxSelected>>", lambda event: self.update_calibration())
        self.update_calibration()

        # OCR frequence input
        self.frequency_label = ttk.Label(self.parameters_frame, text="OCR frequency [ms between pictures]:")
        self.frequency_label.grid(row=2, column=0, padx=5, pady=5)
        self.frequency_entry = ttk.Entry(self.parameters_frame)
        self.frequency = 500
        self.frequency_entry.insert(-1, "500")
        self.frequency_entry.bind("<Return>", lambda event: on_frequency_entry())
        self.frequency_entry.grid(row=2, column=1, padx=5, pady=5)

        def on_frequency_entry():
            try:
                self.frequency = int(self.frequency_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Number of seconds between pictures must be integer.")

        # Aruco marker choice dropdown menu
        self.aruco_dict_label = ttk.Label(self.parameters_frame, text="Aruco dictionary:")
        self.aruco_dropdown = ttk.Combobox(self.parameters_frame, value=list(rc.ARUCO_DICT.keys()))
        self.aruco_dropdown.current(0)
        self.selected_aruco = self.aruco_dropdown.get()
        # # Uncomment to reveal in GUI
        # self.aruco_dict_label.grid(row=3, column=0, padx=5, pady=5)
        # self.aruco_dropdown.grid(row=3, column=1, padx=5, pady=5)

        # Aruco size input
        self.aruco_size_label = ttk.Label(self.parameters_frame, text="Aruco marker size [meters]:")
        self.aruco_size_entry = ttk.Entry(self.parameters_frame)
        self.aruco_size = 0.014
        self.aruco_size_entry.insert(-1, "0.014")
        self.aruco_size_entry.bind("<Return>", lambda event: on_aruco_size_entry())
        ## Uncomment to reveal in GUI
        # self.aruco_size_label.grid(row=4, column=0, padx=5, pady=5)
        # self.aruco_size_entry.grid(row=4, column=1, padx=5, pady=5)

        def on_aruco_size_entry():
            try:
                self.aruco_size = float(self.aruco_size_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number for the aruco marker size.")



        ######################## Charuco board parameters ########################

        # Square size input
        self.square_size_label = ttk.Label(self.parameters_frame, text="Board square size [meters]:")
        self.square_size_entry = ttk.Entry(self.parameters_frame)
        self.square_size = 0.02
        self.square_size_entry.insert(-1, "0.02")
        self.square_size_entry.bind("<Return>", lambda event: on_square_size_entry())
        ## Uncomment to reveal in GUI
        # self.square_size_label.grid(row=5, column=0, padx=5, pady=5)
        # self.square_size_entry.grid(row=5, column=1, padx=5, pady=5)

        def on_square_size_entry():
            try:
                self.square_size = float(self.square_size_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number for the board's square size.")

        # Charuco board width and height inputs
        self.charuco_size_label = ttk.Label(self.parameters_frame, text="Board size [w-h]:")
        self.charuco_size_label.grid(row=6, column=0, padx=5, pady=(5,20))
        self.charuco_width_entry = ttk.Entry(self.parameters_frame)
        self.charuco_width = 4
        self.charuco_width_entry.insert(-1, "4")
        self.charuco_width_entry.bind("<Return>", lambda event: on_charuco_width_entry())
        self.charuco_width_entry.bind("<Return>", lambda event: on_charuco_height_entry(), add='+')
        self.charuco_width_entry.grid(row=6, column=1, padx=5, pady=(5,20))
        self.charuco_height_entry = ttk.Entry(self.parameters_frame)
        self.charuco_height = 4
        self.charuco_height_entry.insert(-1, "4")
        self.charuco_height_entry.bind("<Return>", lambda event: on_charuco_height_entry())
        self.charuco_height_entry.bind("<Return>", lambda event: on_charuco_width_entry(), add='+')
        self.charuco_height_entry.grid(row=6, column=2, padx=5, pady=(5,20))

        def on_charuco_width_entry():
            try:
                self.charuco_width = int(self.charuco_width_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number for the board's width.")

        def on_charuco_height_entry():
            try:
                self.charuco_height = int(self.charuco_height_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number for the board's height.")
        
        
        # Video feed
        self.video_label = tk.Label(self.right_frame)
        self.video_label.pack()
        self.show_camera_feed()
        
        self.indicated_surfaces_counter = 0
        self.show_perspective_corrected_feed()



        ######################## ROI parameters ########################

        # Rectangle drawings on canvas
        self.rectangles = []
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # Scrollable container for list of rectangle variables
        self.list_container = tk.Frame(self.right_frame, highlightbackground = "grey", highlightthickness = 1)
        self.list_canvas = tk.Canvas(self.list_container, width=self.right_frame_width)
        scrollbar = ttk.Scrollbar(self.list_container, command=self.list_canvas.yview)
        self.list_frame = tk.Frame(self.list_canvas)
        self.list_frame.bind(
            "<Configure>",
            lambda e: self.list_canvas.configure(
                scrollregion=self.list_canvas.bbox("all")
            )
        )
        self.list_canvas.create_window((0, 0), window=self.list_frame, anchor="nw")
        self.list_canvas.configure(yscrollcommand=scrollbar.set)
        self.list_container.pack(side=tk.TOP, padx=15, pady=15, ipady=10)
        self.list_canvas.pack(side = tk.LEFT, fill = "both", expand= True)
        scrollbar.pack(side = tk.RIGHT, fill = tk.Y)

        # Create labels for each column
        ttk.Label(self.list_frame, text="Nr.").grid(row=0, column=0, padx=(10,5), pady=15)
        ttk.Label(self.list_frame, text="Variable name").grid(row=0, column=1, padx=5, pady=15)
        ttk.Label(self.list_frame, text="Nums only").grid(row=0, column=2, padx=5, pady=15)
        ttk.Label(self.list_frame, text="Detect").grid(row=0, column=3, padx=5, pady=15)
        ttk.Label(self.list_frame, text="Font type").grid(row=0, column=4, padx=5, pady=15)

        # Start/Stop OCR button
        self.start_button = ttk.Button(self.right_frame, text="Start OCR", command=self.toggle_ocr)
        self.start_button.pack(side=tk.BOTTOM, anchor='s', padx=25, pady=15)
        self.ocr_on = False
        self.results = []

        self.master.update()



    def get_current_screen_size(self):
        # Get the coordinates of the window's top-left corner
        x = self.master.winfo_rootx()
        y = self.master.winfo_rooty()

        monitors = screeninfo.get_monitors()
        monitor = None
        for m in reversed(monitors):
            if m.x <= x <= m.width + m.x and m.y <= y <= m.height + m.y:
                monitor = m
        if monitor is None:
            monitor = monitors[0]

        return monitor.width, monitor.height

    def adjust_frame_sizes(self, event):

        if event.widget != self.master:
            return
        
        # Get screen width and height
        screen_width, screen_height = self.get_current_screen_size()

        # Adjust sizes based on screen dimensions
        self.canvas_max_width = screen_width - self.right_frame_width
        self.canvas_max_height = screen_height - self.left_frame_parameter_height

        self.left_frame.config(width=self.canvas_max_width)

    def get_available_cameras(self):
        # print("get_available_cameras")

        cam_type = self.cam_type.get()
        available_cameras = []

        if cam_type == 'daheng':
            dev_num, dev_info_list = self.device_manager.update_device_list()
            
            # If no Daheng camera found revert to webcam
            if dev_num == 0:
                print("Number of Daheng devices found is 0.Switching to webcam.")
                self.cam = None
                self.cam_type.set("webcam")
                self.selected_cam_type = self.cam_type.get()
                return self.get_available_cameras()

            for cam in dev_info_list:
                # print(cam['sn'])
                available_cameras.append(cam['sn'])

        else:
            index = 0
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            while cap.isOpened():
                available_cameras.append(index)
                cap.release()
                index += 1
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        return available_cameras

    def update_cam_input(self):
        
        print(f"Changing camera to input {self.camera_input_dropdown.get()}")

        cam_type = self.cam_type.get()

        if cam_type == 'daheng':
            
            try:
                if self.cam is not None:
                    self.cam.stream_off()
                    self.cam.close_device()
                self.cam = self.device_manager.open_device_by_sn(self.camera_input_dropdown.get())
                
                self.cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
                self.cam.ExposureTime.set(self.exposure * 1000)
                # self.cam.BalanceWhiteAuto.set(1)
                self.cam.Gain.set(self.gain)

                # get param of improving image quality
                if self.cam.GammaParam.is_readable():
                    gamma_value = self.cam.GammaParam.get()
                    self.gamma_lut = gx.Utility.get_gamma_lut(gamma_value)
                else:
                    self.gamma_lut = None
                if self.cam.ContrastParam.is_readable():
                    contrast_value = self.cam.ContrastParam.get()
                    self.contrast_lut = gx.Utility.get_contrast_lut(contrast_value)
                else:
                    self.contrast_lut = None
                if self.cam.ColorCorrectionParam.is_readable():
                    self.color_correction_param = self.cam.ColorCorrectionParam.get()
                else:
                    self.color_correction_param = 0

                self.cam.data_stream[0].set_acquisition_buffer_number(1)
                self.cam.stream_on()

                raw_image = self.cam.data_stream[0].get_image()
                if raw_image is None:
                    raise Exception("FrameNotRead")
                
                numpy_image = raw_image.get_numpy_array()
                
                self.selected_camera_input = self.camera_input_dropdown.get()
                
                height, width = numpy_image.shape
                if self.calib_w != width or self.calib_h != height:
                    messagebox.showwarning("Warning", "Target and actual resolutions differ.\n"
                                        + "Make sure camera input and camera name correspond.\n"
                                        + f"Target resolution = {self.calib_w}x{self.calib_h}\n"
                                        + f"Actual resolution = {width}x{height}")
                    
            except Exception as e:
                if str(e) == "FrameNotRead":
                    messagebox.showerror("Error", "Unable to retrieve video feed from this camera.\n"
                                        + "Please check the connection and make sure the camera is not being used by another application.")
                else:
                    messagebox.showerror("Error", "An error occurred while trying to retrieve video feed from this camera.")
                    # raise e
                self.camera_input_dropdown.current(self.selected_camera_input)
                self.update_cam_input()
            
        else:

            try :
                if self.cam is not None:
                    self.cam.release()
                self.cam = cv2.VideoCapture(int(self.camera_input_dropdown.get()))
                self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.calib_w)
                self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.calib_h)
                ret,frame = self.get_frame()
                if not ret:
                    raise Exception("FrameNotRead")
                width = self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
                self.selected_camera_input = int(self.camera_input_dropdown.get())

                if self.calib_w != int(width) or self.calib_h != int(height):
                    messagebox.showwarning("Warning", "Target and actual resolutions differ.\n"
                                        + "Make sure camera input and camera name correspond.\n"
                                        + f"Target resolution = {self.calib_w}x{self.calib_h}\n"
                                        + f"Actual resolution = {int(width)}x{int(height)}")

            except Exception as e:
                if str(e) == "FrameNotRead":
                    messagebox.showerror("Error", "Unable to retrieve video feed from this camera.\n"
                                        + "Please check the connection and make sure the camera is not being used by another application.")
                else:
                    messagebox.showerror("Error", "An error occurred while trying to retrieve video feed from this camera.")
                    raise e
                self.camera_input_dropdown.current(self.selected_camera_input)
                self.update_cam_input()

    def update_calibration(self):
        # print(f"Updating calibration to {self.calibration_name_dropdown.get()}")
        # Path to pickle file
        calib_file_path = "cam_calibration/cameras/" + self.calibration_name_dropdown.get() + "/calibration_params.pickle"

        # Load the calibration parameters from the pickle file
        with open(calib_file_path, 'rb') as f:
            calibration_params = pickle.load(f)

        # Extract the parameters from the dictionary
        self.mtx = calibration_params["mtx"]
        self.dist = calibration_params["dist"]
        self.calib_w = int(calibration_params["calib_w"])
        self.calib_h = int(calibration_params["calib_h"])

        if any(x is None for x in (self.mtx, self.dist)):
            messagebox.showerror("Error", "Failed to retrieve calibration parameters.")
            return
        
        if not isinstance(self.mtx, np.ndarray):
            self.mtx = np.array(self.mtx) 

        if not isinstance(self.dist, np.ndarray):
            self.dist = np.array(self.dist) 
        
        self.update_cam_input()

    def refresh_cam_inputs(self):
        # print("refresh_cam_inputs")
        self.camera_inputs = self.get_available_cameras()
        self.camera_input_dropdown['values'] = self.camera_inputs
        self.camera_input_dropdown.current(0)
        print("Current: ", self.camera_input_dropdown.get())

    def update_cam_type(self):
        # print("update_cam_type: ", self.selected_cam_type)

        if self.selected_cam_type == "daheng":
            self.cam.stream_off()
            self.cam.close_device()
            self.daheng_settings_frame.grid_remove()
        else:
            self.cam.release()
            self.daheng_settings_frame.grid(row=1, column=0, columnspan = 3)

        try:
            self.cam = None
            self.refresh_cam_inputs()
            self.update_cam_input()
            self.update_calibration()
            self.selected_cam_type = self.cam_type.get()
            # self.daheng_settings_frame.grid(row=1, column=0, columnspan = 3)
        except Exception as e:
            print("Exception handling")
            self.cam_type.set(self.selected_cam_type)
            self.update_cam_type()
            raise e

    def get_frame(self):

        cam_type = self.cam_type.get()

        if cam_type == 'daheng':
            ret = True
            raw_image = self.cam.data_stream[0].get_image()
            rgb_image = raw_image.convert("RGB")
            rgb_image.image_improvement(self.color_correction_param, self.contrast_lut, self.gamma_lut)
            frame = rgb_image.get_numpy_array()
            if frame is None:
                ret = False

        else:
            ret, frame = self.cam.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return ret, frame

    def show_camera_feed(self):

        self.master.update()

        ret, frame = self.get_frame()
        if not ret:
            self.video_label.after(50, self.show_camera_feed)
            return

        # Get the width and height of the actual image
        height, width = frame.shape[:2]
        
        # Calculate the scale factor to keep the aspect ratio and limit the height to 300
        scale = min(1, 300 / height)
        
        # Resize the original image
        self.resize_width = int(width * scale)
        self.resize_height = int(height * scale)
            
        aruco_dict = cv2.aruco.getPredefinedDictionary(rc.ARUCO_DICT[self.aruco_dropdown.get()])


        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict)

        # Draw A square around the markers
        if np.all(ids is not None): 
            for i in range(len(ids)):
                # get marker corners
                pts = np.int32(corners[i][0])
                # draw lines between corners
                cv2.line(frame, tuple(pts[0]), tuple(pts[1]), (0, 255, 0), thickness=8)
                cv2.line(frame, tuple(pts[1]), tuple(pts[2]), (0, 255, 0), thickness=8)
                cv2.line(frame, tuple(pts[2]), tuple(pts[3]), (0, 255, 0), thickness=8)
                cv2.line(frame, tuple(pts[3]), tuple(pts[0]), (0, 255, 0), thickness=8)

        frame_resized = cv2.resize(frame, (self.resize_width, self.resize_height), interpolation=cv2.INTER_AREA)
        
        img = Image.fromarray(frame_resized)
        # Convert image to PhotoImage
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        
        # Repeat after an interval to capture continuously
        self.video_label.after(50, self.show_camera_feed)

    def toggle_ocr(self):

        # Start OCR
        if not self.ocr_on:

            # Setup database tables
            self.measurement_name_entry_window()
            self.master.wait_window(self.meas_info_window)

            # Check if measurement name has been succesfully set
            if self.meas_name is None:
                return
            
            roi_list = self.create_roi_list()
            
            # Verify that all entries have variable names which are different and not empty
            variables = [roi['variable'] for roi in roi_list]
            if len(set(variables)) != len(variables) or not all(variables):
                messagebox.showerror("Error", "All variables need names, and they need to be different.")
                return

            self.ocr_on = True
            self.start_button.config(text="Stop OCR")
            
            # Block creation of new ROIs during OCR
            self.canvas.unbind("<ButtonPress-1>")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")

            # Start OCR function thread
            meas_comment = self.meas_name_com.get()
            self.ocr_thread = threading.Thread(target=self.call_ocr, daemon=True, args=[roi_list, variables, self.meas_name, meas_comment])
            self.ocr_thread.start()

        # Stop OCR
        else:
            self.ocr_on = False
            self.ocr_thread.join()
            self.start_button.config(text="Start OCR")

            # Allow creation of ROIs again
            self.canvas.bind("<ButtonPress-1>", self.on_button_press)
            self.canvas.bind("<B1-Motion>", self.on_move_press)
            self.canvas.bind("<ButtonRelease-1>", self.on_button_release)       

    def measurement_name_entry_window(self):
        '''Create a new window to enter the name of the measurement (for database)'''

        self.meas_info_window = tk.Toplevel(self.master)
        self.meas_info_window.geometry = ("600x300")
        self.meas_info_window.title("Enter Measurement Information")

        # Create a StringVar to store the measurement name and comment
        self.meas_name_var = tk.StringVar()
        self.meas_name_com = tk.StringVar()
        
        # Used to check if name input was successful in toggle_ocr
        self.meas_name = None

        # Create a label and entry field for measurement name
        meas_name_container = ttk.Frame(self.meas_info_window)
        meas_name_container.pack(padx=10, pady=10)
        meas_name_label = ttk.Label(meas_name_container, text="Measurement Name:")
        meas_name_label.pack(side=tk.LEFT, padx=10, pady=10)
        meas_name_entry = ttk.Entry(meas_name_container, textvariable=self.meas_name_var)
        meas_name_entry.pack(side=tk.LEFT, padx=10, pady=10)

        # Create a label and entry field for measurment comment
        meas_comment_container = ttk.Frame(self.meas_info_window)
        meas_comment_container.pack(padx=10, pady=10)
        meas_comment_label = ttk.Label(meas_comment_container, text="Measurement Comment:")
        meas_comment_label.pack(side=tk.LEFT, padx=10, pady=10)
        meas_comment_entry = ttk.Entry(meas_comment_container, textvariable=self.meas_name_com)
        meas_comment_entry.pack(side=tk.LEFT, padx=10, pady=10)

        def check_and_submit():

            # Create database object to look up name
            db = database.setup_database()

            # Check if measurement with this name already exists in database
            db.cursor.execute("SELECT COUNT(*) FROM Measurements WHERE measurement_name=?", (self.meas_name_var.get(),))
            count = db.cursor.fetchone()[0]
            if count > 0:
                messagebox.showwarning("Warning", "Measurement name already exists. Please enter a different name.")
            else:
                self.meas_name = self.meas_name_var.get()
                self.meas_info_window.destroy()

        # Submit button calls check_and_submit
        submit_button = ttk.Button(self.meas_info_window, text="Submit", command=check_and_submit)
        submit_button.pack(padx=10, pady=10)

    def call_ocr(self, roi_list, variables, meas_name, meas_comment):
        
        # Create database (object cannot be used outside of threaad where it has been created)
        db = database.setup_database()
 
        # Setup database tables
        measurement_id, variable_ids = db.setup_measurement(meas_name, meas_comment, variables)

        # Boolean to show and save video of ROIs if wished
        save_video = False
        display_rois = True
        stacked_roi_images = []
        
        # Initializing steps of the different involved OCR engines
        ocr_engines = set([roi['font'].ocr_engine for roi in roi_list])
        ocr_code.initialize_ocr_engines(ocr_engines)

        self.last_call_time = time.time()
        
        cv2.namedWindow("ROIs", cv2.WINDOW_NORMAL)

        while self.ocr_on:

            frame = copy.copy(self.corrected_frame)

            # Calculations to compensate for OCR execution time in frequency  
            elapsed_time = time.time() - self.last_call_time
            next_call_time = self.frequency/1000 - elapsed_time
            if next_call_time > 0:
                time.sleep(next_call_time)
            
            # Call OCR function
            self.last_call_time = time.time()
            timestamp, values = ocr_on_roi(frame, roi_list)
            # print("process_webcam_feed time = ", time.time() - self.last_call_time)

            # Save results to database
            for value, var_id in zip(values, variable_ids):
                db.insert_frame_data(measurement_id, timestamp, var_id['ID'], value)

            stacked_roi_images.append(self.show_rois(display_rois, frame, roi_list, values))
            
        if display_rois:
            cv2.destroyAllWindows()
        if save_video:
            save_frames_to_avi(stacked_roi_images, meas_name)

    def show_rois(self, display_rois, frame, roi_list, values):

        # Display ROIs
        roi_images = []
        cropped_rois = []
        final_rois = []
        max_width = 0

        # Find the maximum width among the ROI images
        for roi in roi_list:
            x1 = min(roi['ROI'][0],roi['ROI'][2])
            x2 = max(roi['ROI'][0],roi['ROI'][2])
            y1 = min(roi['ROI'][1],roi['ROI'][3])
            y2 = max(roi['ROI'][1],roi['ROI'][3])
            roi_img = frame[y1:y2, x1:x2]
            roi_images.append(roi_img)
            roi_width = roi_img.shape[1]
            max_width = max(max_width, roi_width)

        for roi_img, roi, value in zip(roi_images, roi_list, values):
            cropped_roi, roi_img = crop_roi(roi_img, roi['detection_method'])
            cropped_roi = roi['font'].proc_pipeline(cropped_roi)
            if cropped_roi.shape[2] < 3:
                cropped_roi = cv2.cvtColor(cropped_roi, cv2.COLOR_GRAY2RGB)
            cropped_rois.append(cropped_roi)

            # Give border to original roi if not biggest roi width
            roi_width = roi_img.shape[1]
            if roi_width < max_width:
                border_right = (max_width - roi_width) // 2
                border_left = max_width - roi_width - border_right
                roi_img = cv2.copyMakeBorder(roi_img, 0, 0, border_left, border_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            
            
            # Add the detected text on the ROI
            # Here (0, roi_img.shape[0]) will place the text on the bottom left of the ROI.
            cv2.putText(roi_img, value, (0, roi_img.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            final_rois.append(roi_img)

            # Give border to cropped roi to match biggest roi width
            cropped_width = cropped_roi.shape[1]
            border_right = (max_width - cropped_width) // 2
            border_left = max_width - cropped_width - border_right
            cropped_roi = cv2.copyMakeBorder(cropped_roi, 0, 0, border_left, border_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            final_rois.append(cropped_roi)

        # Display ROIs in a new window
        stacked_roi_img = cv2.vconcat(final_rois)
        stacked_roi_img = cv2.cvtColor(stacked_roi_img, cv2.COLOR_RGB2BGR)
        # print("Stacked images: ", len(stacked_roi_images))
        
        if display_rois:
            cv2.imshow("ROIs", stacked_roi_img)
            cv2.waitKey(1)

        return stacked_roi_img



    ######################## Perspective corrected canvas functions ########################

    def show_perspective_corrected_feed(self):
                
        ret, frame = self.get_frame()
        if not ret:
            self.canvas.after(50, self.show_perspective_corrected_feed)
            return
        
        # Get video feed resolution
        height, width = frame.shape[:2]
        # print(f"Rectified original frame resolution {width}x{height}")
        
        # If target surface info is know, get corrected perspective image
        if self.surface is not None:
            self.corrected_frame, self.last_rvec, self.last_tvec = rc.correct_perspective(frame, self.surface, self.mtx, self.dist, self.last_rvec, self.last_tvec)
            # Store size of corrected perspective frame 
            self.new_height, self.new_width = self.corrected_frame.shape[:2]
            # Get new canvas size with corrected image ratio
            self.new_canvas_width, self.new_canvas_height = resize_with_ratio(self.canvas_max_width, self.canvas_max_height, self.surface.width, self.surface.height)
            frame = copy.copy(self.corrected_frame)
        
        else:
            # Get new canvas size with normal image ratio
            self.new_canvas_width, self.new_canvas_height = resize_with_ratio(self.canvas_max_width, self.canvas_max_height, width, height)
            self.corrected_frame = None

        # Resize the canvas with the new dimensions 
        self.canvas.config(width=self.new_canvas_width, height=self.new_canvas_height)

        # Resize image with new canvas dimensions
        resized_frame = cv2.resize(frame, (self.new_canvas_width, self.new_canvas_height), interpolation=cv2.INTER_AREA)

        # Display image in canvas
        image = Image.fromarray(resized_frame)
        # Photo needs to be stored as GUI attribute or gets garbage collected (I think)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.itemconfig(self.canvas_image, image=self.photo)

        # Repeat after an interval to display continuously
        self.canvas.after(50, self.show_perspective_corrected_feed)

    def create_roi_list(self):

        active_rois = [roi for roi in self.rectangles if roi.isactive]
        roi_frame_coordinates = []
        
        # Adjust ROI coordinates from current canvas size to current actual frame size
        for i, roi in enumerate(active_rois):
            adjusted_x1 = int(roi.coordinates[0] * self.new_width / self.new_canvas_width)
            adjusted_y1 = int(roi.coordinates[1] * self.new_height / self.new_canvas_height)
            adjusted_x2 = int(roi.coordinates[2] * self.new_width / self.new_canvas_width)
            adjusted_y2 = int(roi.coordinates[3] * self.new_height / self.new_canvas_height)
            adjusted_coordinates = (adjusted_x1, adjusted_y1, adjusted_x2, adjusted_y2)
            roi_frame_coordinates.append(adjusted_coordinates)
        
        roi_list = [{
            'variable': roi.variable.get(), 
            'ROI': roi_coords, 
            'only_numbers': roi.only_numbers.get(), 
            'detection_method': roi.detection_dropdown.get(),
            'font': img_processing.fonts[roi.font_type_dropdown.current()]} 
            for roi, roi_coords in zip(active_rois, roi_frame_coordinates)]
        
        return roi_list

    def on_button_press(self, event):
        
        canvas_width = self.canvas.winfo_width()-9
        canvas_height = self.canvas.winfo_height()-9

        self.start_x = self.canvas.canvasx(event.x)
        if self.start_x < 0:
            self.start_x = 0
        if self.start_x > canvas_width:
            self.start_x = canvas_width
        
        self.start_y = self.canvas.canvasy(event.y)
        if self.start_y < 0:
            self.start_y = 0
        if self.start_y > canvas_height:
            self.start_y = canvas_height

        # create a rectangle with initial coordinates
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=1)

    def on_move_press(self, event):
        
        # update the coordinates of the rectangle as the user drags the mouse
        canvas_width = self.canvas.winfo_width()-9
        canvas_height = self.canvas.winfo_height()-9

        cur_x = self.canvas.canvasx(event.x)
        if cur_x < 0:
            cur_x = 0
        if cur_x > canvas_width:
            cur_x = canvas_width
        cur_y = self.canvas.canvasy(event.y)
        if cur_y < 0:
            cur_y = 0
        if cur_y > canvas_height:
            cur_y = canvas_height
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        
        # save the coordinates of the rectangle in an array
        canvas_width = self.canvas.winfo_width()-9
        canvas_height = self.canvas.winfo_height()-9

        x1 = int(self.start_x)
        y1 = int(self.start_y)
        
        x2 = int(self.canvas.canvasx(event.x))
        if x2 < 0:
            x2 = 0
        if x2 > canvas_width:
            x2 = canvas_width

        y2 = int(self.canvas.canvasy(event.y))
        if y2 < 0:
            y2 = 0
        if y2 > canvas_height:
            y2 = canvas_height

        self.canvas.delete(self.rect)
        rectangle = Rectangle(
            canvas= self.canvas,
            roi_list_container= self.list_frame,
            rect_number= len(self.rectangles), 
            coordinates= (x1, y1, x2, y2)
        )
        self.rectangles.append(rectangle)
 

    #################### Indicate surface window functions ####################

    def indicate_surface_window_init(self):
        
        # If a surface is already known, ask if it should be forgotten
        if self.surface is not None:
            forget_surface = messagebox.askyesno(
                "Forget target surface?", 
                "Warning: A target surface has already been found.\n" + 
                "Would you like to forget it and indicate a new one?\n"
            )
            if not forget_surface:
                return

        # Create new window
        self.indic_surface_window = tk.Toplevel(self.master)
        self.indic_surface_window.state('zoomed') 
        self.indic_surface_window.title("Indicate Target Surface")
        self.surface_polygon = None

        # Display number of indicated surface perspectives
        self.indicated_surfaces_counter = 0
        self.saved_coords_label = ttk.Label(self.indic_surface_window, text=f"Number of indicated surfaces: {self.indicated_surfaces_counter} / 2")
        self.saved_coords_label.pack()

        # Create canvas to display video feed and to draw on
        _, frame = self.get_frame()
        height, width = frame.shape[:2]
        max_width = 1280
        max_height = 720
        # width = self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        # height = self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        resized_width, resized_height = resize_with_ratio(max_width, max_height, width, height)
        self.width_ratio = width / resized_width
        self.height_ratio = height / resized_height
        self.indic_surf_canvas = tk.Canvas(self.indic_surface_window, width=max_height,
                                height=resized_height, bd=2, bg="grey")
        self.indic_surf_canvas_image = self.indic_surf_canvas.create_image(0, 0, anchor=tk.NW)
        self.indic_surf_canvas.pack()
        self.indic_surf_canvas.bind("<Button-1>", self.draw_on_press)
        self.indic_surf_canvas.bind("<Motion>", self.draw_on_move)

        # Create "Confirm surface" button
        confirm_button = tk.Button(self.indic_surface_window, text="Confirm surface", command=self.save_coords)
        confirm_button.pack(side="left", fill= "both",  expand=tk.YES, pady=10)

        # Create "Display calculated surface" button
        show_surface_button = tk.Button(self.indic_surface_window, text="Display calculated surface", command=self.toggle_display_surface)
        show_surface_button.pack(side="left", fill= "both", expand=tk.YES, pady=10)
        self.display_surface_on = False

        # Create "Cancel" button
        cancel_button = tk.Button(self.indic_surface_window, text="Cancel", command=self.indic_surface_window.destroy)
        cancel_button.pack(side="right", fill= "both",  expand=tk.YES, pady=10)

        # Initialize variables
        self.surface = None
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(rc.ARUCO_DICT[self.aruco_dropdown.get()])
        self.board = aruco.CharucoBoard((self.charuco_width, self.charuco_height), self.square_size, self.aruco_size, self.aruco_dict)
        self.move_shape = None
        self.img_coordinates = []
        self.point_coords = []
        self.surface_line_eqs = []
        self.indic_last_rvec = None
        self.indic_last_tvec = None

        self.update_indicate_surface_canvas()

    def update_indicate_surface_canvas(self):
        
        # Force Tkinter event queue to update window (window creation can get stuck in queue otherwise)
        self.indic_surface_window.update_idletasks()
        # self.indic_surface_window.update()

        ret, frame = self.get_frame()
        if not ret:
            self.indic_surf_canvas.after(35, self.update_indicate_surface_canvas)
        height, width = frame.shape[:2]
        # print(f"Indic surface frame resolution {width}x{height}")

        if not ret:
            messagebox.showerror("Error", "No image could be read from the camera")
            return
        
        # Detect markers
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, self.aruco_dict)
        aruco.refineDetectedMarkers(gray, self.board, corners, ids, rejectedImgPoints)
        self.retval = False

        # Estimate Charuco board's pose if markers found
        if np.all(ids is not None):
            charucoretval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners, ids, gray, self.board)
            frame = aruco.drawDetectedCornersCharuco(frame, charucoCorners, charucoIds, (0,255,0))
            self.retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, self.board, self.mtx, self.dist, np.zeros((3, 1)), np.zeros((3, 1)))
        
        # If board found update pose estimation
        if self.retval:
            self.indic_last_rvec = rvec
            self.indic_last_tvec = tvec
        
        # Draw pose estimation on image
        if self.indic_last_rvec is not None and self.indic_last_tvec is not None:
            frame = cv2.drawFrameAxes(frame, self.mtx, self.dist, self.indic_last_rvec, self.indic_last_tvec, 0.1)

        # Display the calculated projection of the surface on the image if button has been clicked
        if self.display_surface_on:
            self.display_surface()

        # Resize image to fit in canvas with same ratio
        max_width = 1280
        max_height = 720
        resized_width, resized_height = resize_with_ratio(max_width, max_height, width, height)
        resized_img = cv2.resize(frame, (resized_width, resized_height), interpolation= cv2.INTER_AREA)

        # Update image in canvas
        img = Image.fromarray(resized_img)
        # Photo needs to be stored as GUI attribute or gets garbage collected (I think)
        self.indic_surf_canvas_imgtk = ImageTk.PhotoImage(image=img)
        self.indic_surf_canvas.itemconfig(self.indic_surf_canvas_image, image=self.indic_surf_canvas_imgtk)
        self.indic_surf_canvas.config(width=resized_width, height=resized_height)
        
        # Repeat function
        if self.indic_surface_window.winfo_exists():
            self.indic_surf_canvas.after(35, self.update_indicate_surface_canvas)

    def draw_on_press(self, event):
        
        # Delete new shape if it was already finished
        if len(self.img_coordinates) == 4:
            self.img_coordinates = []
            self.indic_surf_canvas.delete(self.finished_shape)
            return

        # Store new point's image coordinates
        self.img_coordinates.append((event.x, event.y))
        
        # Draw polygon if forth corner entered
        if len(self.img_coordinates) == 4:
            self.finished_shape = self.indic_surf_canvas.create_polygon(self.img_coordinates, outline='red', width=2)

    def draw_on_move(self, event):
        
        # Delete previously drawn shape if it exists
        if self.move_shape:
            self.indic_surf_canvas.delete(self.move_shape)
        
        # Draw new shape
        if len(self.img_coordinates) <4:
            self.move_shape = self.indic_surf_canvas.create_polygon(self.img_coordinates,
                                                    event.x, event.y,
                                                    outline='red', width=2)
        
        # Forces tkinter event queue to update (more responsive to user input)
        self.indic_surface_window.update_idletasks()

    def save_coords(self):

        # Check that the surface rectangle is finished (4 corners indicated) before saving
        if len(self.img_coordinates) != 4:
            messagebox.showerror("Error", "Surface should have four corners", parent= self.indic_surface_window)
            return
        
        # Check if a Charuco pose has ever been estimated
        elif self.indic_last_rvec is None and self.indic_last_tvec is None:
            messagebox.showerror("Error", "No pose estimation for a Charuco board could be calculated", parent= self.indic_surface_window)
            return

        # If pose could not be estimated in this frame, give option to use last estimated pose
        elif not self.retval:
            last_pose_accepted = messagebox.askyesno(
                "Accept last pose estimation?", 
                "Warning: No Charuco pose estimation could be calculated for this frame.\n" + 
                "Would you like to use the last pose estimation calculated?\n" +
                "If so, make sure that the last position of the Charuco corresponds to the displayed pose."
            )
            if not last_pose_accepted:
                return
        
        # Update displayed number of saved surfaces
        self.indicated_surfaces_counter += 1
        self.saved_coords_label.config(text=f"Number of saved surfaces: {self.indicated_surfaces_counter} / 2")
        
        # Translate the coordinates of the canvas pixels to the coordinates of the pixels on the original frame
        self.img_coordinates = [(int(x * self.width_ratio), int(y * self.height_ratio)) for x,y in self.img_coordinates]

        # Store line equeations for each point
        line_eq = [rc.get_line_equation(point, self.mtx, self.indic_last_rvec, self.indic_last_tvec) for point in self.img_coordinates]
        self.surface_line_eqs.append(line_eq)
        
        # If surface has been indicated with enough perspectives, calculate its world coordinates and create surface object
        if self.indicated_surfaces_counter >= 2:
            world_coords = []
            for point in range(4):
                # Get all line equations for one point
                point_lines_w_coords = [surface_line[point] for surface_line in self.surface_line_eqs]
                # Find world coordinates for that point
                point_world_coords = rc.get_point_world_coords(point_lines_w_coords)
                world_coords.append(point_world_coords)
            self.surface = rc.Surface(self.aruco_dict, self.board, world_coords)

        # Delete polygon
        self.indic_surf_canvas.delete(self.finished_shape)
        self.img_coordinates = []

    def toggle_display_surface(self):
        
        # Check if surface object has been created
        if self.surface is None:
            messagebox.showerror("Error", "Indicate surface at least twice.", parent= self.indic_surface_window)
            return
        
        # Delete displayed surface if button is pressed a second time
        if self.display_surface_on:
            self.indic_surf_canvas.delete(self.surface_polygon)
            self.surface_polygon = None
        
        # Toggle display boolean
        self.display_surface_on = not self.display_surface_on

    def display_surface(self):
        
        # Remove polygon being indicated if present
        if self.surface_polygon is not None:
            self.indic_surf_canvas.delete(self.surface_polygon)

        # Get image coordinates of surface corners
        surface_img_coords = [rc.get_point_img_coordinates(point_coords, self.indic_last_rvec, self.indic_last_tvec, self.mtx, self.dist) for point_coords in self.surface.world_coords]

        # Transform coordinates of the point for the canvas scale
        surface_canvas_coords = []
        for point in surface_img_coords:
            point_canvas_coords = int(point[0] / self.width_ratio), int(point[1] / self.height_ratio)
            surface_canvas_coords.append(point_canvas_coords)

        # Create polygon to show position of surface on image 
        self.surface_polygon = self.indic_surf_canvas.create_polygon(surface_canvas_coords, outline='green', width=3)
  
    #################### Functions for import/export of parameters ####################

    def export_parameters(self):
        '''Function managing exporting settings to pickle file'''
        
        if not self.verify_export_conditions():
            return
        
        # Open file dialog window to choose file location and name
        filename = filedialog.asksaveasfilename(
            defaultextension= '.pickle', 
            filetypes= [('Pickle files', '*.pickle'), ('All files', '*.*')],
            title= 'Export parameters'
        )

        # If user cancels the save dialog
        if not filename:  
            return
        
        print(filename)

        export_content = {
            "Camera type": self.cam_type.get(),
            "Calibration file": self.calibration_name_dropdown.get(),
            "Camera input": self.selected_camera_input,
            "Aruco dictionary": self.selected_aruco,
            "Aruco size": self.aruco_size,
            "Square size": self.square_size,
            "Charuco width": self.charuco_width,
            "Charuco heigt": self.charuco_height,
            "ROI list": self.create_roi_list(),
            "Target surface world coordinates": self.surface.world_coords,
        }
        # save the parameters in a pickle file
        with open(filename, 'wb') as f:
            pickle.dump(export_content, f)

        # print("Exported data to:", filename + ".pickle")

    def verify_export_conditions(self):
        '''Verifies that all settings have been set before exporting them'''

        if self.surface is None:
            messagebox.showerror("Error", "Indicate target surface area before exporting data.")
            return False
        
        if not self.rectangles:
            messagebox.showerror("Error", "Indicate some regions of interest.")
            return False
        
        # Verify that all entries have variable names which are different and not empty
        variables = [rectangle.variable.get() for rectangle in self.rectangles if rectangle.isactive]
        if len(set(variables)) != len(variables) or not all(variables):
            messagebox.showerror("Error", "All variables need names, and they need to be different.")
            return False
        
        return True

    def import_parameters(self):
        
        # Get parameter file
        filename = filedialog.askopenfilename()
        print("Importing parameters from ", filename)
        
        # Load the GUI parameters from the pickle file
        with open(filename, 'rb') as f:
            params = pickle.load(f)
        
        # Camera type
        self.cam_type.set(params["Camera type"])
        self.selected_cam_type = self.cam_type.get()

        # Camera input
        self.camera_input_dropdown.set(params["Camera input"])
        self.selected_camera_input = self.camera_input_dropdown.get()

        # Calibration file
        self.calibration_name_dropdown.set(params["Calibration file"])
        self.update_calibration()

        # # Update camera parameters
        # Exposure time
        # Gain

        ## Update charuco parameters
        # Aruco dictionary
        # Aruco size
        # Square size
        # Charuco width
        # Charuco height

        # Target surface world coordinates
        surface_world_coords = params["Target surface world coordinates"]
        aruco_dict = cv2.aruco.getPredefinedDictionary(rc.ARUCO_DICT[self.aruco_dropdown.get()])
        board = aruco.CharucoBoard((self.charuco_width, self.charuco_height), self.square_size, self.aruco_size, aruco_dict)
        self.surface = rc.Surface(aruco_dict, board, surface_world_coords)
        self.surface.update_surface_dimensions()
        self.new_width, self.new_height = resize_with_ratio(self.calib_w, self.calib_h, self.surface.width, self.surface.height)
        self.new_canvas_width, self.new_canvas_height = resize_with_ratio(self.canvas_max_width, self.canvas_max_height, self.surface.width, self.surface.height)

        ## Update ROIs
        # Delete existing rectangles
        for rectangle in self.rectangles:
            rectangle.delete_rect()

        self.rectangles = []
        rois = params["ROI list"]
        for roi in rois:

            # Adjust rectangle coordinates from frame size to canvas size
            adjuted_x0 = int(roi['ROI'][0] / self.new_width * self.new_canvas_width) 
            adjuted_y0 = int(roi['ROI'][1] / self.new_height * self.new_canvas_height)
            adjuted_x1 = int(roi['ROI'][2] / self.new_width * self.new_canvas_width) +1
            adjuted_y1 = int(roi['ROI'][3] / self.new_height * self.new_canvas_height) +1
            adjusted_coords = (adjuted_x0, adjuted_y0, adjuted_x1, adjuted_y1)

            # Find ID of ROI font
            font_list = [font.name for font in img_processing.fonts]
            font_id = font_list.index(roi['font'].name)

            rectangle = Rectangle(
                canvas= self.canvas,
                roi_list_container= self.list_frame,
                rect_number= len(self.rectangles), 
                coordinates= adjusted_coords,
                variable= roi['variable'],
                only_numbers= roi['only_numbers'],
                detect_id=roi['detection_method'],
                font_id= font_id
            )
            self.rectangles.append(rectangle)





def save_frames_to_avi(frames, meas_name):

    # Change name to better format
    new_name = meas_name.lower().replace(" ", "_")
    video_path = "videos/" + new_name + ".avi"

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    video_out = cv2.VideoWriter(video_path, fourcc, 1.0, (frames[0].shape[1], frames[0].shape[0]))

    # Resize frame to make sure they all have the same shape and write each frame to the video
    frame_count = 0
    for frame in frames:
        frame = cv2.resize(frame, (frames[0].shape[1], frames[0].shape[0]))
        video_out.write(frame)

    # Release the video writer
    video_out.release()
    print("Video saved as ", new_name + ".avi")

    # Open the saved video file and count the frames
    video = cv2.VideoCapture(video_path)
    actual_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()

    print(f"Expected number of frames: {len(frames)}")
    print(f"Actual number of frames: {actual_frame_count}")

def resize_with_ratio(max_width, max_height, width, height):

    # Calculate the aspect ratio of the original image
    aspect_ratio = width / float(height)

    # Calculate the maximum aspect ratio allowed based on the given maximum width and height
    max_aspect_ratio = max_width / float(max_height)

    # If the original aspect ratio is greater than the maximum allowed aspect ratio,
    # then the width should be resized to the maximum width, and the height should be
    # resized accordingly to maintain the aspect ratio.
    if aspect_ratio > max_aspect_ratio:
        resized_width = int(max_width)
        resized_height = int(max_width / aspect_ratio)
    # Otherwise, the height should be resized to the maximum height, and the width should
    # be resized accordingly to maintain the aspect ratio.
    else:
        resized_width = int(max_height * aspect_ratio)
        resized_height = int(max_height)

    # Return the resized width and height as a tuple
    return resized_width, resized_height


root = tk.Tk()
root.state('zoomed') 
root.bind('<Escape>', lambda e: root.quit())
gui = OCR_GUI(root)
root.mainloop()