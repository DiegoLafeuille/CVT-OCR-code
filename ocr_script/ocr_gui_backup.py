import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import cv2
from cv2 import aruco
import gxipy as gx
from PIL import Image, ImageTk
from ocr_code import ocr_on_roi
import img_processing
import ocr_code
import pickle
import csv
import datetime
import time
from sympy import symbols, Eq
import threading
import copy


# Names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

class OCR_GUI:

    def __init__(self, master):

        self.ocr_on = False

        # Main window
        self.master = master
        self.master.title("OCR GUI")

        # Left frame
        self.left_frame = tk.Frame(master, width=750)
        self.left_frame.pack(side=tk.LEFT, fill='both', padx=15, pady=15)

        # Right frame
        self.right_frame = tk.Frame(master)
        self.right_frame.pack(side=tk.RIGHT, fill='both', padx=15, pady=15)


        ############################### Left frame ###############################
        
        # Creating radiobuttons to choose camera type and surface detection method
        self.general_params_frame = tk.Frame(self.left_frame, highlightbackground = "grey", highlightthickness = 1)
        self.general_params_frame.pack(side=tk.TOP)

        # Camera type radio buttons
        self.cam_type = tk.StringVar(value="daheng")
        self.cam_type_label = ttk.Label(self.general_params_frame, text="Select type of camera:")
        self.daheng_radio_btn = ttk.Radiobutton(self.general_params_frame, text="Daheng camera", variable=self.cam_type, value="daheng", command=self.update_cam_params)
        self.webcam_radio_btn = ttk.Radiobutton(self.general_params_frame, text="Webcam", variable=self.cam_type, value="webcam", command=self.update_method_params)
        self.cam_type_label.grid(row=0, column=0, padx=30, pady=(10,5))
        self.daheng_radio_btn.grid(row=0, column=1, padx=5, pady=(10,5))
        self.webcam_radio_btn.grid(row=0, column=2, padx=(5,30), pady=(10,5))
        self.device_manager = gx.DeviceManager()
        self.selected_cam_type = self.cam_type.get()

        # Method radio buttons
        self.marker_method = tk.StringVar(value="one_marker")
        self.method_label = ttk.Label(self.general_params_frame, text="Select method to use:")
        self.one_marker_radio_btn = ttk.Radiobutton(self.general_params_frame, text="One-marker method", variable=self.marker_method, value="one_marker", command=self.update_method_params)
        self.four_markers_radio_btn = ttk.Radiobutton(self.general_params_frame, text="Four-marker method", variable=self.marker_method, value="four_markers", command=self.update_method_params)
        self.method_label.grid(row=1, column=0, padx=30, pady=(5,10))
        self.one_marker_radio_btn.grid(row=1, column=1, padx=5, pady=(5,10))
        self.four_markers_radio_btn.grid(row=1, column=2, padx=(5,30), pady=(5,10))
        self.surface_line_eqs = []

        # Image canvas
        self.canvas_max_width = 750
        self.canvas_max_height = 600
        self.canvas = tk.Canvas(self.left_frame, bd=2, bg="grey", width=self.canvas_max_width, height=self.canvas_max_height)
        self.canvas.pack(padx=15, pady=15)
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW)

        # "Indicate display surface" button for single aruco method
        self.indicate_surface_button = ttk.Button(self.left_frame, text="Indicate display surface", command=self.indicate_surface_window_init)
        self.indicate_surface_button.pack(side=tk.BOTTOM, padx=15, pady=5)


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
        refresh_button = ttk.Button(self.parameters_frame, text="Refresh", command=self.refresh_cam_inputs)
        refresh_button.grid(row=0, column=2, padx=5, pady=(20, 5))

        # Camera choice dropdown menu
        self.cam = None
        self.cam_label = ttk.Label(self.parameters_frame, text="Calibration folder:")
        self.cam_label.grid(row=1, column=0, padx=5, pady=5)
        self.camera_names = ["daheng_6mm", "daheng_3mm", "pc08_webcam", "jans_webcam", "diegos_phone", "diegos_iriun", "jans_webcam_charuco"]
        self.camera_name_dropdown = ttk.Combobox(self.parameters_frame, value=self.camera_names)
        self.camera_name_dropdown.current(0)
        self.selected_camera = self.camera_name_dropdown.get()
        self.camera_name_dropdown.grid(row=1, column=1, padx=5, pady=5)
        self.camera_name_dropdown.bind("<<ComboboxSelected>>", lambda event: self.update_calibration())
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
        self.aruco_dict_label.grid(row=3, column=0, padx=5, pady=5)
        self.aruco_dropdown = ttk.Combobox(self.parameters_frame, value=list(ARUCO_DICT.keys()))
        self.aruco_dropdown.current(0)
        self.selected_aruco = self.aruco_dropdown.get()
        self.aruco_dropdown.grid(row=3, column=1, padx=5, pady=5)

        # Aruco size input
        self.aruco_size_label = ttk.Label(self.parameters_frame, text="Aruco marker size [meters]:")
        self.aruco_size_label.grid(row=4, column=0, padx=5, pady=5)
        self.aruco_size_entry = ttk.Entry(self.parameters_frame)
        self.aruco_size = 0.014
        self.aruco_size_entry.insert(-1, "0.014")
        self.aruco_size_entry.bind("<Return>", lambda event: on_aruco_size_entry())
        self.aruco_size_entry.grid(row=4, column=1, padx=5, pady=5)

        def on_aruco_size_entry():
            try:
                self.aruco_size = float(self.aruco_size_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number for the aruco marker size.")



        ######################## Charuco board parameters ########################

        # Square size input
        self.square_size_label = ttk.Label(self.parameters_frame, text="Board square size [meters]:")
        self.square_size_label.grid(row=5, column=0, padx=5, pady=5)
        self.square_size_entry = ttk.Entry(self.parameters_frame)
        self.square_size = 0.02
        self.square_size_entry.insert(-1, "0.02")
        self.square_size_entry.bind("<Return>", lambda event: on_square_size_entry())
        self.square_size_entry.grid(row=5, column=1, padx=5, pady=5)

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
        self.show_camera()
        
        self.frame_counter = 0
        self.saved_surfaces_counter = 0
        self.show_rectified_camera()

        # Rectangle drawings on canvas
        self.rectangles = []
        self.rectangles_drawing = []
        self.rect_drawing_labels = []
        self.labels = []
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # Scrollable container for list of rectangle variables
        self.list_container = tk.Frame(self.right_frame, highlightbackground = "grey", highlightthickness = 1)
        self.list_canvas = tk.Canvas(self.list_container, width=500, height=150)
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
        ttk.Label(self.list_frame, text="Only Numerals").grid(row=0, column=2, padx=5, pady=15)
        ttk.Label(self.list_frame, text="Font type").grid(row=0, column=3, padx=5, pady=15)
        self.rect_labels = []
        self.rect_entries = []
        self.only_nums_list = []
        self.rect_char_checkboxs = []
        self.font_dropdowns = []
        self.rect_delete = []

        # Start/Stop OCR button
        self.start_button = ttk.Button(self.right_frame, text="Start OCR", command=self.toggle_ocr)
        self.start_button.pack(side=tk.BOTTOM, anchor='s', padx=25, pady=15)
        self.results = []

        self.master.update()


    def get_available_cameras(self):

        cam_type = self.cam_type.get()
        available_cameras = []

        if cam_type == 'daheng':
            dev_num, dev_info_list = self.device_manager.update_device_list()
            
            # If no Daheng camera found revert to webcam
            if dev_num == 0:
                print("Number of Daheng devices found is 0")
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
                self.cam.ExposureTime.set(100000.0)
                self.cam.Gain.set(10.0)

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
        print(f"Updating calibration to {self.camera_name_dropdown.get()}")
        # Path to pickle file
        calib_file_path = "cam_calibration/cameras/" + self.camera_name_dropdown.get() + "/calibration_params.pickle"

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
        
        self.update_cam_input()

    def refresh_cam_inputs(self):
        self.camera_inputs = self.get_available_cameras()
        self.camera_input_dropdown['values'] = self.camera_inputs

    def update_cam_params(self):

        if self.selected_cam_type == "daheng":
            self.cam.stream_off()
            self.cam.close_device()
        else:
            self.cam.release()

        try:
            self.cam = None
            self.refresh_cam_inputs()
            self.update_cam_input()
            self.update_calibration()
            self.selected_cam_type = self.cam_type.get()
        except Exception as e:
            print("Exception handling")
            self.cam_type.set(self.selected_cam_type)
            raise e

    def update_method_params(self):
        method = self.marker_method.get()
        if method == "one_marker":
            self.indicate_surface_button.pack(side=tk.BOTTOM, padx=15, pady=5)
            self.aruco_size_label.grid_configure(padx=5, pady=5)
            self.aruco_size_entry.grid_configure(padx=5, pady=5)
            self.square_size_label.grid(row=5, column=0, padx=5, pady=5)
            self.square_size_entry.grid(row=5, column=1, padx=5, pady=5)
            self.charuco_size_label.grid(row=6, column=0, padx=5, pady=(5,20))
            self.charuco_width_entry.grid(row=6, column=1, padx=5, pady=(5,20))
            self.charuco_height_entry.grid(row=6, column=2, padx=5, pady=(5,20))
        else:
            self.indicate_surface_button.pack_forget()
            self.aruco_size_label.grid_configure(padx=5, pady=(5,20))
            self.aruco_size_entry.grid_configure(padx=5, pady=(5,20))
            self.square_size_label.grid_remove()
            self.square_size_entry.grid_remove()
            self.charuco_size_label.grid_remove()
            self.charuco_width_entry.grid_remove()
            self.charuco_height_entry.grid_remove()
        
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

    def toggle_ocr(self):

        # Start OCR
        if not self.ocr_on:
            vars = [var.get() for var in self.rect_entries if var is not None]
            # Verify that all entries have variable names which are different and not empty
            if len(set(vars)) != len(vars) or not all(vars):
                messagebox.showerror("Error", "All variables need names, and they need to be different.")
            else:
                self.ocr_on = True
                self.start_button.config(text="Stop OCR")
                self.canvas.unbind("<ButtonPress-1>")
                self.canvas.unbind("<B1-Motion>")
                self.canvas.unbind("<ButtonRelease-1>")

                roi_list = self.create_rois()
                self.ocr_thread = threading.Thread(target=self.call_ocr, daemon=True, args=[roi_list])
                self.ocr_thread.start()

        # Stop OCR
        else:
            self.ocr_on = False
            self.ocr_thread.join()
            self.start_button.config(text="Start OCR")
            self.canvas.bind("<ButtonPress-1>", self.on_button_press)
            self.canvas.bind("<B1-Motion>", self.on_move_press)
            self.canvas.bind("<ButtonRelease-1>", self.on_button_release)       

    def call_ocr(self, roi_list):

        # Boolean to save video of ROIs if wished
        save_video = False
        
        # Initializing steps of the different involved OCR engines
        ocr_engines = set([roi['font'].ocr_engine for roi in roi_list])
        ocr_code.initialize_ocr_engines(ocr_engines)

        self.last_call_time = time.time()
    
        # Create column names
        cols = ['Timestamp'] + [roi['variable'] for roi in roi_list]

        # Create the csv file and write the headers
        with open('results.csv', mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=cols)
                writer.writeheader()
        
        cv2.namedWindow("ROIs", cv2.WINDOW_NORMAL)

        if save_video:
            stacked_roi_images = []

        while self.ocr_on:


            frame = copy.copy(self.rectified_frame)

            # Calculations to compensate for OCR execution time in frequency  
            elapsed_time = time.time() - self.last_call_time
            next_call_time = self.frequency/1000 - elapsed_time
            if next_call_time > 0:
                time.sleep(next_call_time)
        
            # Indicate in OCR's result file if not all necessary markers have been detected
            if frame is None:
                self.last_call_time = time.time()
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                with open('results.csv', mode='a', newline='') as file:
                    file.write(f"{timestamp},Not enough Aruco markers detected\n")
            
            # Call OCR function if all necessary markers have been detected 
            else:
                self.last_call_time = time.time()
                ocr_on_roi(self.rectified_frame, roi_list, cols)
                # print("process_webcam_feed time = ", time.time() - self.last_call_time)

                roi_images = []
                max_width = 0

                # Find the maximum height among the ROI images
                for roi in roi_list:
                    x1 = min(roi['ROI'][0],roi['ROI'][2])
                    x2 = max(roi['ROI'][0],roi['ROI'][2])
                    y1 = min(roi['ROI'][1],roi['ROI'][3])
                    y2 = max(roi['ROI'][1],roi['ROI'][3])
                    roi_img = frame[y1:y2, x1:x2]
                    roi_width = roi_img.shape[1]
                    max_width = max(max_width, roi_width)

                for roi in roi_list:
                    x1 = min(roi['ROI'][0],roi['ROI'][2])
                    x2 = max(roi['ROI'][0],roi['ROI'][2])
                    y1 = min(roi['ROI'][1],roi['ROI'][3])
                    y2 = max(roi['ROI'][1],roi['ROI'][3])
                    roi_img = frame[y1:y2, x1:x2]

                    roi_img = roi['font'].proc_pipeline(roi_img)
                    if len(roi_img.shape) < 3:
                        roi_img = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2RGB)
                    roi_width = roi_img.shape[1]
                    if roi_width < max_width:
                        border_right = (max_width - roi_width) // 2
                        border_left = max_width - roi_width - border_right
                        roi_img = cv2.copyMakeBorder(roi_img, 0, 0, border_left, border_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    roi_images.append(roi_img)

                # Display ROIs in a new window
                stacked_roi_img = cv2.vconcat(roi_images)
                stacked_roi_img = cv2.cvtColor(stacked_roi_img, cv2.COLOR_RGB2BGR)
                if save_video:
                    stacked_roi_images.append(stacked_roi_img)
                    print(len(stacked_roi_images))
                cv2.imshow("ROIs", stacked_roi_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()
        if save_video:
            save_frames_to_avi(stacked_roi_images)

    def create_rois(self):

        # Create list of ROIs
        rois = []
        # Adjust ROI coordinates from canvas size to frame size
        for i, rectangle in enumerate(self.rectangles):
            if rectangle is None:
                rois.append(None)
                continue

            adjusted_x1 = int(rectangle[0] * self.new_width / self.new_canvas_width)
            adjusted_y1 = int(rectangle[1] * self.new_height / self.new_canvas_height)
            adjusted_x2 = int(rectangle[2] * self.new_width / self.new_canvas_width)
            adjusted_y2 = int(rectangle[3] * self.new_height / self.new_canvas_height)
            roi = (adjusted_x1, adjusted_y1, adjusted_x2, adjusted_y2)

            rois.append(roi)
        
        roi_list = [{'variable': var.get(), 'ROI': roi, 'only_nums': only_nums.get(), 'font': img_processing.fonts[font_type.current()]} for var, roi, only_nums, font_type in zip(self.rect_entries, rois, self.only_nums_list, self.font_dropdowns) if roi is not None]
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
        self.rectangles_drawing.append(self.rect)

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

        self.rectangles.append((x1, y1, x2, y2))
        self.canvas.coords(self.rect, x1, y1, x2, y2)

        # create a new label for the rectangle number
        rect_number = len(self.rectangles)
        label_x = (x1 + x2) // 2
        label_y = y1 - 15
        label = ttk.Label(self.canvas, text=str(rect_number), font=('Arial', 12), background='white', foreground='black')
        label.place(x=label_x, y=label_y, anchor='center')
        self.rect_drawing_labels.append(label)

        self.add_variable_list_elem(rect_number)

    def add_variable_list_elem(self, rect_number):

        # Declaring string variable for storing variable name
        rect_var=tk.StringVar()

        # Creating a label for name using widget Label
        rect_label = ttk.Label(self.list_frame, text=str(rect_number), font=('calibre',10, 'bold'))
        self.rect_labels.append(rect_label)

        # Creating an entry for input name using widget Entry
        rect_entry = ttk.Entry(self.list_frame,textvariable = rect_var, font=('calibre',10,'normal'))
        self.rect_entries.append(rect_entry)

        # Creating a checkbox to ask if only numerals are expected as characters
        only_nums = tk.BooleanVar(value=True)
        rect_char_checkbox = ttk.Checkbutton(self.list_frame, variable=only_nums)
        self.rect_char_checkboxs.append(rect_char_checkbox)
        self.only_nums_list.append(only_nums)

        # Creating a dropdown to choose the font type
        self.font_type_dropdown = ttk.Combobox(self.list_frame, value=[font.name for font in img_processing.fonts])
        self.font_type_dropdown.current(0)
        self.font_dropdowns.append(self.font_type_dropdown)

        # Creating delete button
        delete_btn = ttk.Button(self.list_frame,text = 'Delete', command=lambda: self.delete_rect(rect_number, delete_btn))
        self.rect_delete.append(delete_btn)

        # Placing the label and entry in the required position using grid method
        self.rect_labels[-1].grid(row=rect_number,column=0)
        self.rect_entries[-1].grid(row=rect_number,column=1)
        self.rect_char_checkboxs[-1].grid(row=rect_number,column=2)
        self.font_dropdowns[-1].grid(row=rect_number,column=3)
        self.rect_delete[-1].grid(row=rect_number,column=4)

    def delete_rect(self, rect_number, btn):
    # Deletes a rectangle and its list element if respective button clicked

        rect_id = self.rectangles_drawing[rect_number-1]
        self.canvas.delete(rect_id)
        label_id = self.rect_drawing_labels[rect_number-1]
        label_id.destroy()
        self.rectangles[rect_number-1] = None
        self.rect_labels[rect_number-1].destroy()
        self.rect_entries[rect_number-1].destroy()
        self.rect_entries[rect_number-1] = None
        self.rect_char_checkboxs[rect_number-1].destroy()
        self.only_nums_list[rect_number-1] = None
        self.font_dropdowns[rect_number-1].destroy()
        btn.destroy()

    def show_camera(self):

        self.master.update()

        ret, frame = self.get_frame()
        if not ret:
            self.video_label.after(50, self.show_camera)
            return

        # Get the width and height of the actual image
        height, width = frame.shape[:2]
        
        # Calculate the scale factor to keep the aspect ratio and limit the height to 300
        scale = min(1, 200 / height)
        
        # Resize the original image
        self.resize_width = int(width * scale)
        self.resize_height = int(height * scale)
            
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[self.aruco_dropdown.get()])


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
        self.video_label.after(50, self.show_camera)
 
    def show_rectified_camera(self):
                
        ret, frame = self.get_frame()
        if not ret:
            self.canvas.after(50, self.show_rectified_camera)
            return
        
        # Get video feed resolution
        height, width = frame.shape[:2]
        # print(f"Rectified original frame resolution {width}x{height}")

        # Detect markers in the frame
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[self.aruco_dropdown.get()])
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        corners, self.ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

        warp_input_pts = []

        method = self.marker_method.get()
        
        # Warp with one marker method
        if method == "one_marker":

            board = aruco.CharucoBoard((self.charuco_width, self.charuco_height), self.square_size, self.aruco_size, aruco_dict)
            aruco.refineDetectedMarkers(gray, board, corners, self.ids, rejectedImgPoints)
            
            # If there are markers found by detector
            if np.all(self.ids is not None):
                
                charucoretval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners, self.ids, gray, board)
                frame = aruco.drawDetectedCornersCharuco(frame, charucoCorners, charucoIds, (0,255,0))
                retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, self.mtx, self.dist, np.zeros((3, 1)), np.zeros((3, 1)))
                
                if retval and self.saved_surfaces_counter >= 2:

                    self.surface_img_coords = [self.find_img_coords(point_coords, rvec, tvec) for point_coords in self.surface_world_coords]

                    #  Updating detected dimensions of object every 50 consecutive frames where surface is found
                    if self.frame_counter % 50 == 0:
                        self.new_width, self.new_height = self.get_surface_dims_one_marker()
                    self.frame_counter += 1
                    
                    for point in self.surface_img_coords:
                        point_canvas_coords = [int(point[0]), int(point[1])]
                        warp_input_pts.append(point_canvas_coords)
                    warp_input_pts = np.float32(warp_input_pts)

                elif retval:
                    frame = cv2.drawFrameAxes(frame, self.mtx, self.dist, rvec, tvec, 0.1)
            

        # Warp with four markers method
        elif method == "four_markers":

            if np.all(self.ids is not None):
        
                tvecs = []
                rvecs = []
            
                for id in self.ids:  
                    index = np.where(self.ids == id)[0][0]
                    
                    # Estimate pose of each marker and return the values rvec and tvec
                    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[index], self.aruco_size, self.mtx, self.dist)
                    tvecs.append(tvec)
                    rvecs.append(rvec)
        
                if all(id in self.ids for id in [1,2,3,4]):
                
                    roi_corners = {}

                    for id in self.ids:  
                        index = np.where(self.ids == id)[0][0]
                        if id == 1:
                            roi_corners["A"] = [int(corners[index][0][2][0]), int(corners[index][0][2][1])] # Bottom right corner of ID 1
                        if id == 2:
                            roi_corners["B"] = [int(corners[index][0][1][0]), int(corners[index][0][1][1])] # Top right corner of ID 2
                            # roi_corners["D"] = [int(corners[index][0][3][0]), int(corners[index][0][3][1])] # Bottom left corner of ID 2
                        if id == 3:
                            roi_corners["C"] = [int(corners[index][0][0][0]), int(corners[index][0][0][1])] # Top left corner of ID 3
                        if id == 4:
                            roi_corners["D"] = [int(corners[index][0][3][0]), int(corners[index][0][3][1])] # Bottom left corner of ID 4
                            # roi_corners["B"] = [int(corners[index][0][1][0]), int(corners[index][0][1][1])] # Top right corner of ID 4
                    
                    warp_input_pts = np.float32([roi_corners["A"], roi_corners["B"], roi_corners["C"], roi_corners["D"]])

                    #  Updating detected dimensions of object every 50 consecutive frames where surface is found
                    if self.frame_counter % 50 == 0:
                        self.new_width, self.new_height = self.get_surface_dims_four_markers(tvecs, self.ids)
                    self.frame_counter += 1
                
                else:
                    self.frame_counter = 0  
                    # aruco.drawDetectedMarkers(frame, corners)

                    # Draw marker lines and distance
                    for i, corner in enumerate(corners):
                        
                        # Draw marker axes from pose estimation
                        cv2.drawFrameAxes(frame, self.mtx, self.dist, rvecs[i], tvecs[i], 0.1, 3)
                        
                        # Draw lines between corners
                        pts = np.int32(corner[0])
                        cv2.line(frame, tuple(pts[0]), tuple(pts[1]), (0, 255, 0), thickness=3)
                        cv2.line(frame, tuple(pts[1]), tuple(pts[2]), (0, 255, 0), thickness=3)
                        cv2.line(frame, tuple(pts[2]), tuple(pts[3]), (0, 255, 0), thickness=3)
                        cv2.line(frame, tuple(pts[3]), tuple(pts[0]), (0, 255, 0), thickness=3)
                        
                        # Draw distance from camera to marker center
                        cv2.putText(frame, 
                                f"{round(np.linalg.norm(tvecs[i][0][0]), 2)} m", 
                                corner[0][1].astype(int), 
                                cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2, 
                                cv2.LINE_AA,
                        )
                    
                
        # If surface coordinates are found
        if len(warp_input_pts) > 0:
            
            # Compute the perspective transform M and warp frame
            warp_output_pts = np.float32([[0, 0],
                                    [0, self.new_height - 1],
                                    [self.new_width - 1, self.new_height - 1],
                                    [self.new_width - 1, 0]])

            M = cv2.getPerspectiveTransform(warp_input_pts,warp_output_pts)

            self.rectified_frame = cv2.warpPerspective(frame,M,(self.new_width, self.new_height),flags=cv2.INTER_CUBIC)
            # frame_height, frame_width = frame.shape[:2]

            self.new_canvas_width, self.new_canvas_height = resize_with_ratio(self.canvas_max_width, self.canvas_max_height, self.new_width, self.new_height)

            # Convert to RGB format
            frame = copy.copy(self.rectified_frame)
        
        else:

            self.new_canvas_width, self.new_canvas_height = resize_with_ratio(self.canvas_max_width, self.canvas_max_height, width, height)
            
            self.rectified_frame = None

        # Resize image with new dimensions
        resized_frame = cv2.resize(frame, (self.new_canvas_width, self.new_canvas_height), interpolation=cv2.INTER_AREA)

        # Adjust the canvas size to match the image size
        self.canvas.config(width=self.new_canvas_width, height=self.new_canvas_height)

        # Display image in canvas
        image = Image.fromarray(resized_frame)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.itemconfig(self.canvas_image, image=self.photo)

        # Repeat after an interval to capture continuously
        self.canvas.after(50, self.show_rectified_camera)

    def get_surface_dims_one_marker(self):

        width_AD = np.linalg.norm(self.surface_world_coords[0] - self.surface_world_coords[3])
        width_BC = np.linalg.norm(self.surface_world_coords[1] - self.surface_world_coords[2])
        surface_w = max(width_AD, width_BC)

        height_AB = np.linalg.norm(self.surface_world_coords[0] - self.surface_world_coords[1])
        height_CD = np.linalg.norm(self.surface_world_coords[2] - self.surface_world_coords[3])
        surface_h = max(height_AB, height_CD)


        # Resize for max image size within original size while keeping surface ratio
        new_width, new_height = resize_with_ratio(self.calib_w, self.calib_h, surface_w, surface_h)
    
        return new_width, new_height

    def get_surface_dims_four_markers(self, tvecs, ids):
     
        indexA = np.where(ids == 1)[0][0]
        indexB = np.where(ids == 2)[0][0]
        # indexD = np.where(ids == 2)[0][0]
        indexC = np.where(ids == 3)[0][0]
        indexD = np.where(ids == 4)[0][0]
        # indexB = np.where(ids == 4)[0][0]

        width_AD = np.linalg.norm(tvecs[indexA]-tvecs[indexD]) - self.aruco_size
        width_BC = np.linalg.norm(tvecs[indexB]-tvecs[indexC]) - self.aruco_size
        surface_w = max(width_AD, width_BC)

        height_AB = np.linalg.norm(tvecs[indexA]-tvecs[indexB]) - self.aruco_size
        height_CD = np.linalg.norm(tvecs[indexC]-tvecs[indexD]) - self.aruco_size
        surface_h = max(height_AB, height_CD)

        new_width, new_height = resize_with_ratio(self.calib_w, self.calib_h, surface_w, surface_h)
    
        return new_width, new_height
  


    #################### Methods for one marker method ####################

    def indicate_surface_window_init(self):

        self.indic_surface_window = tk.Toplevel(self.master)
        self.indic_surface_window.title("Indicate Target Surface")

        self.surface_polygon = None
        self.saved_surfaces_counter = 0
        self.saved_coords_label = ttk.Label(self.indic_surface_window, text=f"Number of saved surfaces: {self.saved_surfaces_counter} / 2")
        self.saved_coords_label.pack()

        # Create canvas to display video feed
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
        
        # print("in init: ", self.indic_surf_canvas.winfo_width(), self.indic_surf_canvas.winfo_height())
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
        self.move_shape = None
        self.coords = []
        self.point_coords = []
        self.surface_world_coords = []
        self.surface_line_eqs = []

        self.update_indic_surf_canvas()

    def update_indic_surf_canvas(self):

        self.indic_surface_window.update()

        ret, frame = self.get_frame()

        # height, width = frame.shape[:2]
        # print(f"Indic surface frame resolution {width}x{height}")

        if not ret:
            messagebox.showerror("Error", "No image could be read from the camera")
            return
        
        # # Undistort image
        # h,  w = frame.shape[:2]
        # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h))
        # frame = cv2.undistort(frame, self.mtx, self.dist, None, newcameramtx)
        # x, y, w, h = roi
        # frame = frame[y:y+h, x:x+w]

        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[self.aruco_dropdown.get()])
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

        board = aruco.CharucoBoard((self.charuco_width, self.charuco_height), self.square_size, self.aruco_size, aruco_dict)
        aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedImgPoints)

        if np.all(ids is not None):
            charucoretval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners, ids, gray, board)
            frame = aruco.drawDetectedCornersCharuco(frame, charucoCorners, charucoIds, (0,255,0))
            self.retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, self.mtx, self.dist, np.zeros((3, 1)), np.zeros((3, 1)))

            if self.retval == True:
                self.rvec = rvec
                self.tvec = tvec
                frame = cv2.drawFrameAxes(frame, self.mtx, self.dist, self.rvec, self.tvec, 0.1)

                if self.display_surface_on:
                    self.display_surface()

        # Undistort image
        height,  width = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (width,height), 1, (width,height))
        frame = cv2.undistort(frame, self.mtx, self.dist, None, newcameramtx)
        x, y, width, height = roi
        frame = frame[y:y+height, x:x+width]

        max_width = 1280
        max_height = 720
        resized_width, resized_height = resize_with_ratio(max_width, max_height, width, height)
        resized_img = cv2.resize(frame, (resized_width, resized_height), interpolation= cv2.INTER_AREA)

        img = Image.fromarray(resized_img)
        self.indic_surf_canvas_imgtk = ImageTk.PhotoImage(image=img)
        self.indic_surf_canvas.itemconfig(self.indic_surf_canvas_image, image=self.indic_surf_canvas_imgtk)
        self.indic_surf_canvas.config(width=resized_width, height=resized_height)
        
        if self.indic_surface_window.winfo_exists():
            self.indic_surf_canvas.after(35, self.update_indic_surf_canvas)

    def draw_on_press(self, event):
        
        # Restart new shape if it was already finished
        if len(self.coords) == 4:
            self.coords = []
            self.indic_surf_canvas.delete(self.finished_shape)

        self.coords.append((event.x, event.y))
        
        if len(self.coords) == 4:
            self.finished_shape = self.indic_surf_canvas.create_polygon(self.coords, outline='red', width=2)

    def draw_on_move(self, event):
        
        # Delete previously drawn shape if it exists
        if self.move_shape:
            self.indic_surf_canvas.delete(self.move_shape)
        
        # Draw new shape
        if len(self.coords) <4:
            self.move_shape = self.indic_surf_canvas.create_polygon(self.coords,
                                                    event.x, event.y,
                                                    outline='red', width=2)

    def save_coords(self):

        if len(self.coords) != 4:
            messagebox.showerror("Error", "Surface should have four corners", parent= self.indic_surface_window)
            return
        
        if not self.retval:
            messagebox.showerror("Error", "Pose estimation of charuco board could not be calculated", parent= self.indic_surface_window)
            return
        
        self.saved_surfaces_counter += 1
        self.saved_coords_label.config(text=f"Number of saved surfaces: {self.saved_surfaces_counter} / 2")
        
        # Transform the coordinates of the canvas pixel to the original size of the frame
        self.coords = [(int(x * self.width_ratio), int(y * self.height_ratio)) for x,y in self.coords]

        self.line_eq = [self.get_line_equation(point) for point in self.coords]
        self.surface_line_eqs.append(self.line_eq)
        
        if self.saved_surfaces_counter >= 2:
            new_world_coords = []
            for point in range(4):
                # Get all line equations for one point
                point_lines_w_coords = [surface_line[point] for surface_line in self.surface_line_eqs]
                # Find world coordinates for that point
                point_world_coords = self.find_point_world_coords(point_lines_w_coords)
                new_world_coords.append(point_world_coords)
            self.surface_world_coords = new_world_coords

        self.indic_surf_canvas.delete(self.finished_shape)
        self.coords = []
     
    def get_line_equation(self, point):

        self.s = symbols('s')
        x = np.array([[point[0]], [point[1]], [1]])

        rot_mat = cv2.Rodrigues(self.rvec)[0]
        inv_rodr = np.linalg.inv(rot_mat)
        
        inv_mtx = np.linalg.inv(self.mtx)

        line_equation = inv_rodr @ ((inv_mtx @ (self.s * x)) - self.tvec.reshape((3,1)))
        return line_equation

    def toggle_display_surface(self):

        if self.saved_surfaces_counter < 2:
            messagebox.showerror("Error", "Indicate surface at least twice.", parent= self.indic_surface_window)
            return
        
        if self.display_surface_on:
            self.indic_surf_canvas.delete(self.surface_polygon)
            self.surface_polygon = None
        
        self.display_surface_on = not self.display_surface_on

    def display_surface(self):
        
        if self.surface_polygon is not None:
            self.indic_surf_canvas.delete(self.surface_polygon)

        self.surface_img_coords = [self.find_img_coords(point_coords, self.rvec, self.tvec) for point_coords in self.surface_world_coords]
        
        # Transform coordinates of the point for the canvas scale
        surface_canvas_coords = []
        for point in self.surface_img_coords:
            point_canvas_coords = int(point[0] / self.width_ratio), int(point[1] / self.height_ratio)
            surface_canvas_coords.append(point_canvas_coords)

        self.surface_polygon = self.indic_surf_canvas.create_polygon(surface_canvas_coords, outline='green', width=3)
        
    def find_point_world_coords(self, line_eqs):
        """
        Given a list of lines, find the point that is closest to all the lines.
        The least square method is used here.
        """

        # Compute the intersection points of all pairs of lines
        intersections = []
        
        
        for i in range(len(line_eqs)):

            # print([f"{x}= {eq[0]}" for x, eq in zip(["X","Y","Z"], line_eqs[i])])   

            for j in range(i+1, len(line_eqs)):

                # Replace s in each line's system of equation X(s), Y(s), Z(s), to get two points
                p1 = np.array([eq[0].subs(self.s, 0) for eq in line_eqs[i]]).reshape((3))
                p2 = np.array([eq[0].subs(self.s, 1) for eq in line_eqs[i]]).reshape((3))
                
                q1 = np.array([eq[0].subs(self.s, 0) for eq in line_eqs[j]]).reshape((3))
                q2 = np.array([eq[0].subs(self.s, 1) for eq in line_eqs[j]]).reshape((3))
                
                intersection = self.lines_intersection(p1, p2, q1, q2)
                intersections.append(intersection)

        # x = least_squares_average(intersections)
        x = np.mean(intersections, axis=0)
        # x = estimate_3d_coordinates

        # Return the solution as a point
        return x
    
    def lines_intersection(self, p1, p2, q1, q2):
        """
        Given two lines, each represented by a pair of 3D points, compute the coordinates 
        of the middle point of the shortest segment linking the two lines.
        """

        # Calculate direction vectors for each line
        p_dir = p2 - p1
        q_dir = q2 - q1

        # Calculate the translation between the two points of origin
        orig_translation = p1 - q1

        a = np.dot(p_dir, p_dir.T)
        b = np.dot(p_dir, q_dir.T)
        c = np.dot(q_dir, q_dir.T)
        d = np.dot(p_dir, orig_translation.T)
        e = np.dot(q_dir, orig_translation.T)
        denom = a*c - b*b

        if denom != 0:
            s = (b*e - c*d) / denom
            t = (a*e - b*d) / denom
            result = 0.5 * (p1 + s*p_dir + q1 + t*q_dir)
            result = [float(x) for x in result]
            return result
        
        # If lines are parallel
        else:
            return np.nan * np.ones(3)

    def find_img_coords(self, world_coords, rvec, tvec):
        
        point_3d = np.array(world_coords).reshape((1, 1, 3))

        # Use projectPoints to project the 3D point onto the 2D image plane
        point_2d, _ = cv2.projectPoints(point_3d, rvec, tvec, self.mtx, self.dist)

        # Extract the pixel coordinates of the projected point
        pixel_coords = tuple(map(int, point_2d[0, 0]))

        return pixel_coords

def save_frames_to_avi(frames):

    print(f"Frames in video: {len(frames)}")
    # Define the codec and create a VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out = cv2.VideoWriter('roi_video.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 1.0, (frames[0].shape[1], frames[0].shape[0]))

    # Write each frame to the video
    for frame in frames:
        video_out.write(frame)

    # Release the video writer
    video_out.release()

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
root.bind('<Escape>', lambda e: root.quit())
root.geometry = ("1080x720")
gui = OCR_GUI(root)
root.mainloop()