import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import cv2
from cv2 import aruco
import numpy as np
from PIL import Image, ImageTk
from ocr_code import process_webcam_feed
import pickle
import csv
import easyocr
import datetime
from sympy import symbols, Eq


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
        self.ocr_count = 0
        self.reader = easyocr.Reader(['en'], gpu=False)

        # Main window
        self.master = master
        self.master.title("OCR GUI")

        # Left frame
        self.left_frame = tk.Frame(master, bg='grey', width=750)
        self.left_frame.pack(side=tk.LEFT, fill='both', padx=15, pady=15)

        # Right frame
        self.right_frame = tk.Frame(master, bg='grey', width=300)
        self.right_frame.pack(side=tk.RIGHT, fill='both', padx=15, pady=15)

        # Image canvas
        self.canvas_max_width = 750
        self.canvas_max_height = 600
        self.canvas = tk.Canvas(self.left_frame, bd=2, bg="grey", width=self.canvas_max_width, height=self.canvas_max_height)
        self.canvas.pack(side=tk.TOP, padx=15, pady=15)
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW)

        # "Find surface" button for single aruco method
        self.indicate_surface_button = ttk.Button(self.left_frame, text="Indicate display surface", command=self.indicate_surface_window)
        self.indicate_surface_button.pack(side=tk.BOTTOM, anchor='s', padx=15, pady=15)
        
        # Create new frame to hold the dropdown grid
        self.parameters_frame = tk.Frame(self.right_frame)
        self.parameters_frame.pack(fill=tk.X, padx=5, pady=5)

        # Camera choice dropdown menu
        self.cap = None
        self.cam_label = ttk.Label(self.parameters_frame, text="Camera:")
        self.cam_label.grid(row=1, column=0, padx=5, pady=5)
        self.camera_names = ["jans_webcam_charuco", "jans_webcam", "diegos_phone", "diegos_iriun", "pc08_webcam"]
        self.camera_name_dropdown = ttk.Combobox(self.parameters_frame, value=self.camera_names)
        self.camera_name_dropdown.current(0)
        self.selected_camera = self.camera_name_dropdown.get()
        self.camera_name_dropdown.grid(row=1, column=1, padx=5, pady=5)
        self.camera_name_dropdown.bind("<<ComboboxSelected>>", lambda event: self.update_cam())

        # Camera input choice dropdown menu
        self.cam_input = ttk.Label(self.parameters_frame, text="Camera input:")
        self.cam_input.grid(row=0, column=0, padx=5, pady=5)
        self.camera_inputs = get_available_cameras()
        self.camera_input_dropdown = ttk.Combobox(self.parameters_frame, value=self.camera_inputs)
        self.camera_input_dropdown.current(0)
        self.selected_camera_input = int(self.camera_input_dropdown.get())
        self.camera_input_dropdown.grid(row=0, column=1, padx=5, pady=5)
        self.camera_input_dropdown.bind("<<ComboboxSelected>>", lambda event: self.update_cam_input())
        refresh_button = ttk.Button(self.parameters_frame, text="Refresh", command=self.get_cam_inputs)
        refresh_button.grid(row=0, column=2, padx=5, pady=5)
        self.update_cam()

        # Aruco marker choice dropdown menu
        self.aruco_dict_label = ttk.Label(self.parameters_frame, text="Aruco dictionary:")
        self.aruco_dict_label.grid(row=2, column=0, padx=5, pady=5)
        self.aruco_dropdown = ttk.Combobox(self.parameters_frame, value=list(ARUCO_DICT.keys()))
        self.aruco_dropdown.current(0)
        self.selected_aruco = self.aruco_dropdown.get()
        self.aruco_dropdown.grid(row=2, column=1, padx=5, pady=5)

        # Aruco size input
        self.aruco_size_label = ttk.Label(self.parameters_frame, text="Aruco marker size [meters]:")
        self.aruco_size_label.grid(row=3, column=0, padx=5, pady=5)
        self.aruco_size_entry = ttk.Entry(self.parameters_frame)
        self.aruco_size_entry.insert(-1, "0.08")
        self.aruco_size_entry.grid(row=3, column=1, padx=5, pady=5)
        self.aruco_size = 0.08

        def on_size_entry():
            try:
                self.aruco_size = float(self.aruco_size_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number for the aruco marker size.")

        self.aruco_size_entry.bind("<Return>", lambda event: on_size_entry())

        # OCR frequence input
        self.frequence_label = ttk.Label(self.parameters_frame, text="OCR frequence [seconds between pictures]:")
        self.frequence_label.grid(row=4, column=0, padx=5, pady=5)
        self.frequence_entry = ttk.Entry(self.parameters_frame)
        self.frequence_entry.insert(-1, "1")
        self.frequence_entry.grid(row=4, column=1, padx=5, pady=5)
        self.frequence = 1

        def on_frequence_entry():
            try:
                self.frequence = float(self.frequence_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Number of seconds between pictures must be integer.")

        self.frequence_entry.bind("<Return>", lambda event: on_frequence_entry())

        # Creating a checkbox to choose surface detection method
        self.one_marker_label = ttk.Label(self.parameters_frame, text="Use one marker method:")
        self.one_marker_label.grid(row=5, column=0, padx=5, pady=5)
        self.one_marker = tk.BooleanVar(value=False)
        self.one_marker_checkbox = ttk.Checkbutton(self.parameters_frame, variable=self.one_marker)
        self.one_marker_checkbox.grid(row=5, column=1, padx=5, pady=5)

        # Video feed
        self.cap = cv2.VideoCapture(self.selected_camera_input)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.calib_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.calib_h)
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        scale = min(1, 300 / height)
        
        # Resize the original image
        self.resize_width = int(width * scale)
        self.resize_height = int(height * scale)
        self.video_label = tk.Label(self.right_frame, width=self.resize_width, height=self.resize_height)
        self.video_label.pack()
        self.show_camera()
        
        self.frame_counter = 0
        self.show_rectified_camera()

        # Rectangle drawings on canvas
        self.rectangles = []
        self.rectangles_drawing = []
        self.rect_drawing_labels = []
        self.labels = []
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # Rectangle variable names list
        self.list_frame = tk.Frame(self.right_frame, bg='white', width=300, height=150)
        self.list_frame.pack(side=tk.TOP, fill='x', padx=15, pady=15)
        # Create labels for each column
        ttk.Label(self.list_frame, text="Nr.").grid(row=0, column=0)
        ttk.Label(self.list_frame, text="Variable name").grid(row=0, column=1)
        ttk.Label(self.list_frame, text="Only Numerals").grid(row=0, column=2)
        self.rect_labels = []
        self.rect_entries = []
        self.only_nums_list = []
        self.rect_char_checkboxs = []
        self.rect_delete = []

        # Start/Stop OCR button
        self.start_button = ttk.Button(self.right_frame, text="Start OCR", command=self.toggle_ocr)
        self.start_button.pack(side=tk.BOTTOM, anchor='s', padx=15, pady=15)
        self.results = []


    def update_cam_input(self):
        
        print(f"Changing camera to input {int(self.camera_input_dropdown.get())}")

        try :
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(int(self.camera_input_dropdown.get()))
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.calib_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.calib_h)
            ret,frame = self.cap.read()
            if not ret:
                raise Exception("FrameNotRead")
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
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

    def update_cam(self):
        print(f"Updating camera: {self.camera_name_dropdown.get()}")
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

    def get_cam_inputs(self):
        self.camera_inputs = get_available_cameras()
        self.camera_input_dropdown['values'] = self.camera_inputs

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
        
        # Stop OCR
        else:
            self.ocr_on = False
            self.start_button.config(text="Start OCR")
            self.ocr_count = 0
            self.canvas.bind("<ButtonPress-1>", self.on_button_press)
            self.canvas.bind("<B1-Motion>", self.on_move_press)
            self.canvas.bind("<ButtonRelease-1>", self.on_button_release)       

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
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=3)
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

        # create a new label for the rectangle number
        rect_number = len(self.rectangles)
        label_x = (x1 + x2) // 2
        label_y = y1 - 15
        label = ttk.Label(self.canvas, text=str(rect_number), font=('Arial', 12), background='white', foreground='black')
        label.place(x=label_x, y=label_y, anchor='center')
        self.rect_drawing_labels.append(label)

        # For a scrollable frame for the Label-Entry-Button list, check out:
        # https://blog.teclado.com/tkinter-scrollable-frames/

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

        # Creating delete button
        delete_btn = ttk.Button(self.list_frame,text = 'Delete', command=lambda: self.delete_rect(rect_number, delete_btn))
        self.rect_delete.append(delete_btn)

        # Placing the label and entry in the required position using grid method
        self.rect_labels[-1].grid(row=rect_number,column=0)
        self.rect_entries[-1].grid(row=rect_number,column=1)
        self.rect_char_checkboxs[-1].grid(row=rect_number,column=2)
        self.rect_delete[-1].grid(row=rect_number,column=3)

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
        btn.destroy()

    def show_camera(self):

        # Get the width and height of the actual image
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # Calculate the scale factor to keep the aspect ratio and limit the height to 300
        scale = min(1, 300 / height)
        
        # Resize the original image
        self.resize_width = int(width * scale)
        self.resize_height = int(height * scale)
            
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[self.aruco_dropdown.get()])

        # Get the latest frame and convert into Image
        ret, frame = self.cap.read()
        if not ret:
            self.video_label.after(10, self.show_camera)
            return

        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict)

        # Draw A square around the markers
        # aruco.drawDetectedMarkers(frame, corners) 
        if np.all(ids is not None): 
            for i in range(len(ids)):
                # get marker corners
                pts = np.int32(corners[i][0])
                # draw lines between corners
                cv2.line(frame, tuple(pts[0]), tuple(pts[1]), (0, 255, 0), thickness=3)
                cv2.line(frame, tuple(pts[1]), tuple(pts[2]), (0, 255, 0), thickness=3)
                cv2.line(frame, tuple(pts[2]), tuple(pts[3]), (0, 255, 0), thickness=3)
                cv2.line(frame, tuple(pts[3]), tuple(pts[0]), (0, 255, 0), thickness=3)

        cv2image= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        cv2image_resized = cv2.resize(cv2image, (self.resize_width, self.resize_height))
        
        img = Image.fromarray(cv2image_resized)
        # Convert image to PhotoImage
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        
        # Repeat after an interval to capture continuously
        self.video_label.after(10, self.show_camera)
 
    def show_rectified_camera(self):
                
        # Read a new frame from the camera
        ret, frame = self.cap.read()
        if not ret:  
            self.canvas.after(int(self.frequence*1000), self.show_rectified_camera)
            return
        
        # Get video feed resolution
        width  = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # Detect markers in the frame
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[self.aruco_dropdown.get()])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)


        # If there are markers found by detector
        if np.all(ids is not None):  

            tvecs = []
            rvecs = []
          
            for id in ids:  
                index = np.where(ids == id)[0][0]
                
                # Estimate pose of each marker and return the values rvec and tvec
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[index], self.aruco_size, self.mtx, self.dist)
                tvecs.append(tvec)
                rvecs.append(rvec)
                (rvec - tvec).any()  # get rid of numpy value array error

            # If all four markers detected
            if all(id in ids for id in [1,2,3,4]) and not self.one_marker.get():

                roi_corners = {}

                for id in ids:  
                    index = np.where(ids == id)[0][0]
                    if id == 1:
                        roi_corners["A"] = [int(corners[index][0][2][0]), int(corners[index][0][2][1])] # Bottom right corner of ID 1
                    if id == 2:
                        roi_corners["D"] = [int(corners[index][0][3][0]), int(corners[index][0][3][1])] # Bottom left corner of ID 2
                    if id == 3:
                        roi_corners["C"] = [int(corners[index][0][0][0]), int(corners[index][0][0][1])] # Top left corner of ID 3
                    if id == 4:
                        roi_corners["B"] = [int(corners[index][0][1][0]), int(corners[index][0][1][1])] # Top right corner of ID 4

                #  Updating detected dimensions of object every 50 consecutive frames where all 4 markers are detected
                if self.frame_counter % 50 == 0:
                    self.new_width, self.new_height = self.get_obj_dims(tvecs, ids)

                self.frame_counter += 1
                
                # Compute the perspective transform M and warp frame
                input_pts = np.float32([roi_corners["A"], roi_corners["B"], roi_corners["C"], roi_corners["D"]])
                output_pts = np.float32([[0, 0],
                                        [0, self.new_height - 1],
                                        [self.new_width - 1, self.new_height - 1],
                                        [self.new_width - 1, 0]])
                M = cv2.getPerspectiveTransform(input_pts,output_pts)

                frame = cv2.warpPerspective(frame,M,(self.new_width, self.new_height),flags=cv2.INTER_LINEAR)
                frame_height, frame_width = frame.shape[:2]

                # Calculate new image dimensions while keeping original ratio
                scale = min(self.canvas_max_width / self.new_width, self.canvas_max_height / self.new_height)
                self.new_canvas_width = int(self.new_width*scale)
                self.new_canvas_height = int(self.new_height*scale)
            
            # If some but not all markers are detected
            else:
                self.frame_counter = 0       

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

                # Calculate new image dimensions while keeping original ratio
                scale = min(self.canvas_max_width / width, self.canvas_max_height / height)
                self.new_canvas_width = int(width*scale)
                self.new_canvas_height = int(height*scale)

        else:
            self.frame_counter = 0

            # Calculate new image dimensions while keeping original ratio
            scale = min(self.canvas_max_width / width, self.canvas_max_height / height)
            self.new_canvas_width = int(width*scale)
            self.new_canvas_height = int(height*scale)

        # Convert to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize image with new dimensions
        resized_frame = cv2.resize(frame, (self.new_canvas_width, self.new_canvas_height), interpolation=cv2.INTER_AREA)

        # Adjust the canvas size to match the image size
        self.canvas.config(width=self.new_canvas_width, height=self.new_canvas_height)

        # Display image in canvas
        image = Image.fromarray(resized_frame)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.itemconfig(self.canvas_image, image=self.photo)
        # Adjust the scroll region to the image size
        self.canvas.config(scrollregion=self.canvas.bbox(self.canvas_image)) 


        # Call OCR function if Start button has been pushed
        if self.ocr_on:

            # Create list of ROIs
            rois = []
            # Adjust ROI coordinates from canvas size to frame size
            for rectangle in self.rectangles:
                if rectangle is None:
                    rois.append(None)
                    continue
                adjusted_x1 = int(rectangle[0] * self.new_width / self.new_canvas_width)
                adjusted_y1 = int(rectangle[1] * self.new_height / self.new_canvas_height)
                adjusted_x2 = int(rectangle[2] * self.new_width / self.new_canvas_width)
                adjusted_y2 = int(rectangle[3] * self.new_height / self.new_canvas_height)
                roi = (adjusted_x1, adjusted_y1, adjusted_x2, adjusted_y2)
                rois.append(roi)
            
            roi_list = [{'variable': var.get(), 'ROI': roi, 'only_nums': only_nums.get()} for var, roi, only_nums in zip(self.rect_entries, rois, self.only_nums_list) if roi is not None]
            
            # Create column names
            cols = ['Timestamp'] + [roi['variable'] for roi in roi_list]
            
            # Create the csv file and write the headers if start button has just been pushed
            if self.ocr_count == 0:
                with open('results.csv', mode='w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=cols)
                    writer.writeheader()
            
            # Indicate in OCR's result file if not all necessary markers have been detected
            if ids is not None and not all(id in ids for id in [1,2,3,4]) or ids is None:
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                with open('results.csv', mode='a', newline='') as file:
                    file.write(f"{timestamp},Not enough Aruco markers detected\n")
            
            # Call OCR function if all necessary markers have been detected 
            else:
                process_webcam_feed(frame, self.reader, roi_list, cols)
            
            self.ocr_count +=1

        # Repeat after an interval to capture continuously
        self.canvas.after(int(self.frequence*1000), self.show_rectified_camera)

    def get_obj_dims(self, tvecs, ids):
     
        indexA = np.where(ids == 1)[0][0]
        indexB = np.where(ids == 4)[0][0]
        indexC = np.where(ids == 3)[0][0]
        indexD = np.where(ids == 2)[0][0]

        width_AB = np.linalg.norm(tvecs[indexA]-tvecs[indexB]) - self.aruco_size
        width_CD = np.linalg.norm(tvecs[indexC]-tvecs[indexD]) - self.aruco_size
        surface_w = max(width_AB, width_CD)

        height_AD = np.linalg.norm(tvecs[indexA]-tvecs[indexD]) - self.aruco_size
        height_BC = np.linalg.norm(tvecs[indexB]-tvecs[indexC]) - self.aruco_size
        surface_h = max(height_AD, height_BC)


        # Resize image
        scale = surface_h / surface_w

        if self.calib_h * scale > self.calib_w:
            new_width = self.calib_w
            new_height = int(self.calib_w / scale)
        else:
            new_height = self.calib_h
            new_width = int(self.calib_h * scale)
    

        return new_width, new_height
  




    # Methods for one marker method

    def indicate_surface_window(self):

        self.window_feed = tk.Toplevel(self.master)
        self.window_feed.title("Indicate Target Surface")

        self.found_surface = None
        self.surface_coords = []
        self.saved_coords_label = ttk.Label(self.window_feed, text=f"Number of saved surfaces: {len(self.surface_coords)} / 2")
        self.saved_coords_label.pack()

        # Create canvas to display video feed
        max_width = 1280
        max_height = 720
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        resized_width, resized_height = resize_with_aspect(max_width, max_height, width, height)
        self.width_ratio = width / resized_width
        self.height_ratio = height / resized_height
        self.indic_surf_canvas = tk.Canvas(self.window_feed, width=resized_width,
                                height=resized_height, bd=2, bg="grey")
        self.indic_surf_canvas_image = self.indic_surf_canvas.create_image(0, 0, anchor=tk.NW)
        self.indic_surf_canvas.pack()

        # Create "Confirm surface" button
        confirm_button = tk.Button(self.window_feed, text="Confirm surface", command=self.save_coords)
        confirm_button.pack(side="left", fill= "both",  expand=tk.YES, pady=10)

        # Create "Display calculated surface" button
        show_surface_button = tk.Button(self.window_feed, text="Display calculated surface", command=self.toggle_display_surface)
        show_surface_button.pack(side="left", fill= "both", expand=tk.YES, pady=10)
        self.display_surface_on = False

        # Create "Cancel" button
        cancel_button = tk.Button(self.window_feed, text="Cancel", command=self.window_feed.destroy)
        cancel_button.pack(side="right", fill= "both",  expand=tk.YES, pady=10)

        self.move_shape = None
        self.indic_surf_canvas.bind("<Button-1>", self.draw_on_press)
        self.indic_surf_canvas.bind("<Motion>", self.draw_on_move)
        self.coords = []
        self.point_coords = []
        self.surface_line_eqs = []

        self.update_indic_surf_canvas(resized_width, resized_height)
    
    def update_indic_surf_canvas(self, resized_width, resized_height):
        
        self.window_feed.update()

        ret, frame = self.cap.read()

        if not ret:
            messagebox.showerror("Error", "No image could be read from the camera")
            return



        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[self.aruco_dropdown.get()])
        squareLength = 0.02
        markerLength = 0.016
        chess_width = 4
        chess_height = 4
        board = aruco.CharucoBoard((chess_width, chess_height), squareLength, markerLength, aruco_dict)


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)
        aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedImgPoints)
        aruco.drawDetectedMarkers(frame, corners)

        # If there are markers found by detector
        if np.all(ids is not None):
            
            charucoretval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners, ids, gray, board)
            frame = aruco.drawDetectedCornersCharuco(frame, charucoCorners, charucoIds, (0,255,0))
            retval, self.rvec, self.tvec = aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, self.mtx, self.dist, np.zeros((3, 1)), np.zeros((3, 1)))
            
            if retval == True:
                frame = cv2.drawFrameAxes(frame, self.mtx, self.dist, self.rvec, self.tvec, 0.01)


        if self.display_surface_on:
            self.display_surface()

        # Convert the frame to PIL Image format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(frame, (resized_width, resized_height))


        # aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[self.aruco_dropdown.get()])
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
        # aruco.drawDetectedMarkers(frame, corners)

        # # If there are markers found by detector
        # if np.all(ids is not None):
            
        #     # Iterate over detected markers and estimate their pose
        #     for id in ids:  
        #         index = np.where(ids == id)[0][0]
        #         if id == 1:
        #             # Estimate pose of each marker and return the values rvec and tvec
        #             self.rvec, self.tvec, _ = aruco.estimatePoseSingleMarkers(corners[index], self.aruco_size, self.mtx, self.dist)
        #             (self.rvec - self.tvec).any()  # get rid of numpy value array error
        #             cv2.drawFrameAxes(frame, self.mtx, self.dist, self.rvec, self.tvec, 0.1, 3)

        # if self.display_surface_on:
        #     self.display_surface()

        # # Convert the frame to PIL Image format
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # resized_img = cv2.resize(frame, (resized_width, resized_height))






        img = Image.fromarray(resized_img)

        # Convert the PIL Image to ImageTk format
        self.indic_surf_canvas_imgtk = ImageTk.PhotoImage(image=img)
        # Update the canvas with the new frame
        self.indic_surf_canvas.itemconfig(self.indic_surf_canvas_image, image=self.indic_surf_canvas_imgtk)
        # Schedule the next update in 33 millisecond = 30fps
        self.indic_surf_canvas.after(33, self.update_indic_surf_canvas, resized_width, resized_height)

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
            messagebox.showerror("Error", "Surface should have four corners", parent= self.window_feed)
            return
            
        self.surface_coords.append(self.coords)
        self.saved_coords_label.config(text=f"Number of saved surfaces: {len(self.surface_coords)} / 2")
        
        # Transform the coordinates of the pixel to the original size of the frame
        self.coords = [(int(x * self.width_ratio), int(y * self.height_ratio)) for x,y in self.coords]

        self.line_eq = [self.get_line_equation(point) for point in self.coords]

        self.surface_line_eqs.append(self.line_eq)

        self.indic_surf_canvas.delete(self.finished_shape)
        self.coords = []
     
    def get_line_equation(self, point):

        self.s = symbols('s')
        x = np.array([[point[0]], [point[1]], [1]])

        rot_mat = cv2.Rodrigues(self.rvec)[0]
        inv_rodr = np.linalg.inv(rot_mat)
        
        inv_mtx = np.linalg.inv(self.mtx)

        # step1 = inv_mtx @ (self.s * x)
        # step2 = step1 - self.tvec.reshape((3,1))
        # step3 = inv_rodr @ step2
        # line_equation = step3

        line_equation = inv_rodr @ ((inv_mtx @ (self.s * x)) - self.tvec.reshape((3,1)))
        # print (f"line equation shape:\n{line_equation.shape}")
        # print (f"line equation:\n{line_equation}")
        return line_equation

    def toggle_display_surface(self):

        if len(self.surface_line_eqs) < 2:
            messagebox.showerror("Error", "Indicate surface at least twice.", parent= self.window_feed)
            return
        
        if self.display_surface_on:
            self.indic_surf_canvas.delete(self.found_surface)
            self.found_surface = None
        else:
            self.surface_world_coords = []
            for point in range(4):
                # Get all line equations for one point
                point_lines_w_coords = [surface_line[point] for surface_line in self.surface_line_eqs]
                # Find world coordinates for that point
                point_world_coords = self.find_point_world_coords(point_lines_w_coords)
                self.surface_world_coords.append(point_world_coords)
        
        self.display_surface_on = not self.display_surface_on

    def display_surface(self):
        
        if self.found_surface is not None:
            try:
                self.indic_surf_canvas.delete(self.found_surface)
            except:
                print("Problem by found_surface deletion: ", self.found_surface)

        surface_img_coords = [self.find_img_coords(point_coords) for point_coords in self.surface_world_coords]
        self.found_surface = self.indic_surf_canvas.create_polygon(surface_img_coords, outline='green', width=3)
        
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
        Given two lines, each represented by a pair of 3D points, compute their
        intersection point.
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

    def find_img_coords(self, world_coords):
        
        point_3d = np.array(world_coords).reshape((1, 1, 3))

        # Use projectPoints to project the 3D point onto the 2D image plane
        point_2d, _ = cv2.projectPoints(point_3d, self.rvec, self.tvec, self.mtx, self.dist)

        # Extract the pixel coordinates of the projected point
        pixel_coords = tuple(map(int, point_2d[0, 0]))

        # Transform coordinates of the point for the canvas scale
        pixel_coords = int(pixel_coords[0] / self.width_ratio), int(pixel_coords[1] / self.height_ratio)

        return pixel_coords





def get_available_cameras():
    available_cameras = []
    index = 0
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    while cap.isOpened():
        available_cameras.append(index)
        ret, frame = cap.read()
        cap.release()
        index += 1
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return available_cameras

def resize_with_aspect(max_width, max_height, width, height):

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
root.config(bg="skyblue")
gui = OCR_GUI(root)
root.mainloop()




# def estimate_3d_coordinates(points, num_iterations=100, sample_size=3, threshold=0.1):
#     best_model = None
#     best_inliers = []
    
#     for _ in range(num_iterations):
#         # Randomly select a minimal sample
#         sample_indices = np.random.choice(len(points), size=sample_size, replace=False)
#         sample_points = points[sample_indices]
        
#         # Fit a model (e.g., plane or sphere) to the sample points
        
#         # Evaluate the model and find inliers
#         residuals = calculate_residuals(points, sample_points, best_model)
#         inliers = np.where(residuals < threshold)[0]
        
#         # Check if this model has more inliers than the previous best model
#         if len(inliers) > len(best_inliers):
#             best_model = fit_model(points[inliers])
#             best_inliers = inliers
            
#     # Refit the model using all inliers
#     final_model = fit_model(points[best_inliers])
    
#     # Return the estimated 3D coordinates
#     estimated_coordinates = least_squares_estimation(points[best_inliers], final_model)
#     print(np.shape(estimated_coordinates))
    
#     return estimated_coordinates

# def calculate_residuals(points, sample_points, model):
#     # Calculate residuals between the model and all points
#     residuals = np.abs(distance_to_model(points, sample_points, model))
#     return residuals

# def fit_model(data):
#     """Fits a line model to the given 2D data points using least squares."""
#     x = data[:, 0]
#     y = data[:, 1]
#     A = np.vstack([x, np.ones_like(x)]).T
#     m, c = np.linalg.lstsq(A, y, rcond=None)[0]
#     return m, c

# def distance_to_model(data, model):
#     """Calculates the perpendicular distance from each data point to the line model."""
#     m, c = model
#     x = data[:, 0]
#     y = data[:, 1]
#     distances = np.abs(m * x - y + c) / np.sqrt(m**2 + 1)
#     return distances

# def least_squares_estimation(data, num_iterations, threshold):
#     """Performs RANSAC least squares estimation to robustly fit a line model to the data."""
#     best_model = None
#     best_inliers = None
#     best_num_inliers = 0

#     for i in range(num_iterations):
#         # Randomly sample two points from the data
#         sample_indices = np.random.choice(data.shape[0], 2, replace=False)
#         sample = data[sample_indices]

#         # Fit a model to the sampled points
#         model = fit_model(sample)

#         # Calculate the distances from all points to the model
#         distances = distance_to_model(data, model)

#         # Count the number of inliers (points within the threshold)
#         inliers = distances < threshold
#         num_inliers = np.count_nonzero(inliers)

#         # Check if this model is the best one so far
#         if num_inliers > best_num_inliers:
#             best_model = model
#             best_inliers = inliers
#             best_num_inliers = num_inliers

#     # Refit the model using all the inliers
#     inlier_points = data[best_inliers]
#     best_model = fit_model(inlier_points)

#     return best_model

# def least_squares_average(points, n_outliers=0.2, max_iterations=100, tolerance=1e-6):
#     """Computes the least squares estimate of the average point for a list of 3D points.

#     Args:
#         points: A list of 3D points.
#         n_outliers: The percentage of distances to discard as outliers.
#         max_iterations: The maximum number of iterations to perform.
#         tolerance: The tolerance for convergence.

#     Returns:
#         The least squares estimate of the average point.
#     """
#     # Convert points to a numpy array for easier computation.
#     points = np.array(points)

#     # Choose an initial estimate for the average point.
#     average = np.mean(points, axis=0)

#     for iteration in range(max_iterations):
#         # Compute the distance between each point and the current estimate.
#         distances = np.linalg.norm(points - average, axis=1)

#         # Sort the distances in ascending order.
#         sorted_distances = np.sort(distances)

#         # Discard the top n percent of the distances as outliers.
#         n = int(n_outliers * len(points))
#         inliers = sorted_distances[n:]

#         # Compute the least squares estimate of the average point using the remaining distances.
#         if len(inliers) == 0:
#             # All points were outliers, so we can't compute a least squares estimate.
#             break

#         new_average = np.mean(points[distances <= inliers[-1]], axis=0)

#         # Check for convergence.
#         if np.allclose(average, new_average, atol=tolerance):
#             break

#         average = new_average

#     return average
