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
        
        
        # Create new frame to hold the dropdown grid
        self.parameters_frame = tk.Frame(self.right_frame)
        self.parameters_frame.pack(fill=tk.X, padx=5, pady=5)

        # Camera choice dropdown menu
        self.cam_label = tk.Label(self.parameters_frame, text="Camera:")
        self.cam_label.grid(row=1, column=0, padx=5, pady=5)
        self.camera_names = ["jans_webcam", "diegos_phone", "diegos_iriun"]
        self.camera_name_dropdown = ttk.Combobox(self.parameters_frame, value=self.camera_names)
        self.camera_name_dropdown.current(0)
        self.update_cam()
        self.selected_camera = self.camera_name_dropdown.get()
        self.camera_name_dropdown.grid(row=1, column=1, padx=5, pady=5)
        self.camera_name_dropdown.bind("<<ComboboxSelected>>", lambda event: self.update_cam())

        # Camera channel choice dropdown menu
        self.ch_label = tk.Label(self.parameters_frame, text="Camera channel:")
        self.ch_label.grid(row=0, column=0, padx=5, pady=5)
        self.camera_channels = get_available_cameras()
        self.camera_ch_dropdown = ttk.Combobox(self.parameters_frame, value=self.camera_channels)
        self.camera_ch_dropdown.current(0)
        self.selected_camera_ch = int(self.camera_ch_dropdown.get())
        self.camera_ch_dropdown.grid(row=0, column=1, padx=5, pady=5)
        self.camera_ch_dropdown.bind("<<ComboboxSelected>>", lambda event: self.update_cam_ch())
        self.cap = None
        self.update_cam_ch()

        # Aruco marker choice dropdown menu
        self.aruco_dict_label = tk.Label(self.parameters_frame, text="Aruco dictionary:")
        self.aruco_dict_label.grid(row=2, column=0, padx=5, pady=5)
        self.aruco_dropdown = ttk.Combobox(self.parameters_frame, value=list(ARUCO_DICT.keys()))
        self.aruco_dropdown.current(0)
        self.selected_aruco = self.aruco_dropdown.get()
        self.aruco_dropdown.grid(row=2, column=1, padx=5, pady=5)

        # Aruco size input
        self.aruco_size_label = tk.Label(self.parameters_frame, text="Aruco marker size [meters]:")
        self.aruco_size_label.grid(row=3, column=0, padx=5, pady=5)
        self.aruco_size_entry = tk.Entry(self.parameters_frame)
        self.aruco_size_entry.insert(-1, "0.04")
        self.aruco_size_entry.grid(row=3, column=1, padx=5, pady=5)
        self.aruco_size = 0.04

        def on_size_entry():
            try:
                self.aruco_size = float(self.aruco_size_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number for the aruco marker size.")

        self.aruco_size_entry.bind("<Return>", lambda event: on_size_entry())

        # Video feed
        self.cap = cv2.VideoCapture(self.selected_camera_ch)
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
        self.rect_labels = []
        self.rect_entries = []
        self.rect_delete = []

        # Start/Stop OCR button
        self.start_button = tk.Button(self.right_frame, text="Start OCR", command=self.toggle_ocr)
        self.start_button.pack(side=tk.BOTTOM, anchor='s', padx=15, pady=15)
        self.results = []




    def update_cam_ch(self):
        print(f"Changing camera to channel: {self.calib_w}x{self.calib_h}")
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(int(self.camera_ch_dropdown.get()))
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.calib_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.calib_h)

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
        print(f"{self.calib_w}x{self.calib_h}")

        if any(x is None for x in (self.mtx, self.dist)):
            messagebox.showerror("Error", "Failed to retrieve calibration parameters.")

    def toggle_ocr(self):
        if self.ocr_on:
            self.ocr_on = False
            self.start_button.config(text="Start OCR")
            self.ocr_count = 0
            self.canvas.bind("<ButtonPress-1>", self.on_button_press)
            self.canvas.bind("<B1-Motion>", self.on_move_press)
            self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        else:
            self.ocr_on = True
            self.start_button.config(text="Stop OCR")
            self.canvas.unbind("<ButtonPress-1>")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")

    def on_button_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

        # create a rectangle with initial coordinates
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=3)
        self.rectangles_drawing.append(self.rect)

    def on_move_press(self, event):
        # update the coordinates of the rectangle as the user drags the mouse
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        # save the coordinates of the rectangle in an array
        x1 = int(self.start_x)
        y1 = int(self.start_y)
        x2 = int(self.canvas.canvasx(event.x))
        y2 = int(self.canvas.canvasy(event.y))
        self.rectangles.append((x1, y1, x2, y2))

        # create a new label for the rectangle number
        rect_number = len(self.rectangles)
        label_x = (x1 + x2) // 2
        label_y = y1 - 15
        label = tk.Label(self.canvas, text=str(rect_number), font=('Arial', 12), bg='white', fg='black')
        label.place(x=label_x, y=label_y, anchor='center')
        self.rect_drawing_labels.append(label)

        # deletes a rectangle if respective button clicked
        def delete_rect(rect_number, btn):
            rect_id = self.rectangles_drawing[rect_number-1]
            self.canvas.delete(rect_id)
            label_id = self.rect_drawing_labels[rect_number-1]
            label_id.destroy()
            self.rectangles[rect_number-1] = None
            self.rect_labels[rect_number-1].destroy()
            self.rect_entries[rect_number-1].destroy()
            btn.destroy()


        # For a scrollable frame for the Label-Entry-Button list, check out:
        # https://blog.teclado.com/tkinter-scrollable-frames/

        # declaring string variable for storing variable name
        rect_var=tk.StringVar()

        # creating a label for name using widget Label
        rect_label = tk.Label(self.list_frame, text=str(rect_number), font=('calibre',10, 'bold'))
        self.rect_labels.append(rect_label)
        # creating a entry for input name using widget Entry
        rect_entry = tk.Entry(self.list_frame,textvariable = rect_var, font=('calibre',10,'normal'))
        self.rect_entries.append(rect_entry)

        # creating delete 
        delete_btn = tk.Button(self.list_frame,text = 'Delete', command=lambda: delete_rect(rect_number, delete_btn))
        self.rect_delete.append(delete_btn)

        # placing the label and entry in the required position using grid method
        self.rect_labels[-1].grid(row=rect_number-1,column=0, sticky='w')
        self.rect_entries[-1].grid(row=rect_number-1,column=1, sticky='w'+'e')
        self.rect_delete[-1].grid(row=rect_number-1,column=2, sticky='e')

    def show_camera(self):

        if self.selected_camera_ch != int(self.camera_ch_dropdown.get()):
            self.selected_camera_ch = int(self.camera_ch_dropdown.get())
            self.cap.release()

            # Start a new video capture with the selected camera
            self.cap = cv2.VideoCapture(int(self.camera_ch_dropdown.get()))
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.calib_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.calib_h)

            # Get the width and height of the original image
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
        # Detect markers in the frame
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
            return
        
        # Get video feed resolution
        width  = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # print(f"Resolution = {int(width)}x{int(height)}")
        
        # Detect markers in the frame
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[self.aruco_dropdown.get()])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

        # If there are markers found by detector
        if np.all(ids is not None):  

            roi_corners = {}
            tvecs = []
            rvecs = []
            
            # Iterate over detected markers and estimate their pose
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

                # Estimate pose of each marker and return the values rvec and tvec
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[index], self.aruco_size, self.mtx, self.dist)
                tvecs.append(tvec)
                rvecs.append(rvec)
                (rvec - tvec).any()  # get rid of numpy value array error

            # If all four markers detected
            if all(id in ids for id in [1,2,3,4]):

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
                self.new_width = int(width*scale)
                self.new_height = int(height*scale)
                        
        else:
            self.frame_counter = 0

            # Calculate new image dimensions while keeping original ratio
            scale = min(self.canvas_max_width / width, self.canvas_max_height / height)
            self.new_width = int(width*scale)
            self.new_height = int(height*scale)

        # Convert to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize image with new dimensions
        frame = cv2.resize(frame, (self.new_width, self.new_height), interpolation=cv2.INTER_AREA)
        # print(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")

        # Adjust the canvas size to match the image size
        self.canvas.config(width=self.new_width, height=self.new_height)

        # Display image in canvas
        image = Image.fromarray(frame)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.itemconfig(self.canvas_image, image=self.photo)
        # Adjust the scroll region to the image size
        self.canvas.config(scrollregion=self.canvas.bbox(self.canvas_image)) 


        # Call OCR function if Start button has been pushed
        if self.ocr_on:

            # Create list of ROIs and column names
            roi_list = [{'variable': var.get(), 'ROI': rectangle} for var, rectangle in zip(self.rect_entries, self.rectangles) if rectangle is not None]
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
        self.canvas.after(330, self.show_rectified_camera)

    def get_obj_dims(self, tvecs, ids):
     
        indexA = np.where(ids == 1)[0][0]
        indexB = np.where(ids == 4)[0][0]
        indexC = np.where(ids == 3)[0][0]
        indexD = np.where(ids == 2)[0][0]

        width_AB = np.linalg.norm(tvecs[indexA]-tvecs[indexB]) - self.aruco_size
        width_CD = np.linalg.norm(tvecs[indexC]-tvecs[indexD]) - self.aruco_size
        corners_width = max(width_AB, width_CD)

        height_AD = np.linalg.norm(tvecs[indexA]-tvecs[indexD]) - self.aruco_size
        height_BC = np.linalg.norm(tvecs[indexB]-tvecs[indexC]) - self.aruco_size
        corners_height = max(height_AD, height_BC)


        # Resize image
        scale = corners_height / corners_width

        if self.canvas_max_height * scale > self.canvas_max_width:
            new_width = self.canvas_max_width
            new_height = int(self.canvas_max_width / scale)
        else:
            new_height = self.canvas_max_height
            new_width = int(self.canvas_max_height * scale)
    

        return new_width, new_height
    
        # # Repeat after an interval to capture continuously
        # self.canvas.after(330, self.show_rectified_camera)
  

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


root = tk.Tk()
root.bind('<Escape>', lambda e: root.quit())
root.geometry = ("1080x720")
root.config(bg="skyblue")
gui = OCR_GUI(root)
root.mainloop()
