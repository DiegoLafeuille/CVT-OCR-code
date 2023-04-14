import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from ocr_code import process_webcam_feed

class OCR_GUI:

    def __init__(self, master):

        # Main window
        self.master = master
        self.master.title("OCR GUI")

        # Left frame
        self.left_frame = tk.Frame(master, bg='grey', width=750)
        self.left_frame.pack(side=tk.LEFT, fill='both', padx=15, pady=15)

        # Right frame
        self.right_frame = tk.Frame(master, bg='grey', width=300)
        self.right_frame.pack(side=tk.RIGHT, fill='both', padx=15, pady=15)

        # Define the dictionary to use
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)

        # Image canvas
        self.canvas_max_width = 750
        self.canvas_max_height = 600
        self.canvas = tk.Canvas(self.left_frame, bd=2, bg="grey", width=self.canvas_max_width, height=self.canvas_max_height)
        self.canvas.pack(side=tk.TOP, padx=15, pady=15)
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW)
        
        # Camera choice dropdown menu
        self.camera_names = get_available_cameras()
        self.camera_dropdown = ttk.Combobox(self.right_frame, value = self.camera_names)
        self.camera_dropdown.current(1)
        self.selected_camera = int(self.camera_dropdown.get())
        self.camera_dropdown.pack(side=tk.TOP, padx=15, pady=15)

        # Video feed
        self.cap = cv2.VideoCapture(self.selected_camera)
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        scale = min(1, 300 / height)
        
        # Resize the original image
        self.resize_width = int(width * scale)
        self.resize_height = int(height * scale)
        self.video_label = tk.Label(self.right_frame, width=self.resize_width, height=self.resize_height)
        self.video_label.pack()
        self.show_camera()
        
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
        self.list_frame = tk.Frame(self.right_frame, bg='white', width=300, height=300)
        self.list_frame.pack(side=tk.TOP, fill='x', padx=15, pady=15)
        self.rect_labels = []
        self.rect_entries = []
        self.rect_delete = []

        # Start OCR button
        self.button = tk.Button(self.right_frame, text="Start OCR", command=self.start_ocr)
        self.button.pack(side=tk.BOTTOM, anchor='s', padx=15, pady=15)
        self.results = []

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
 
    def show_rectified_camera(self):

        if self.selected_camera != int(self.camera_dropdown.get()):
            self.selected_camera = int(self.camera_dropdown.get())
            self.cap.release()
            self.cap = cv2.VideoCapture(int(self.camera_dropdown.get()))
            
        ret, frame = self.cap.read()  # read a new frame from the webcam
        if not ret:  # if reading fails
            return

        # Detect markers in the frame
        corners, ids, _ = cv2.aruco.detectMarkers(frame, self.dictionary)

        roi_corners = {}

        # Draw detected markers and their IDs on the frame
        if ids is not None:

            for i, id in enumerate(ids):
                if id == 1:
                    roi_corners["A"] = [int(corners[i][0][2][0]), int(corners[i][0][2][1])] # Bottom right corner of ID 1
                if id == 2:
                    roi_corners["D"] = [int(corners[i][0][3][0]), int(corners[i][0][3][1])] # Bottom left corner of ID 2
                if id == 3:
                    roi_corners["C"] = [int(corners[i][0][0][0]), int(corners[i][0][0][1])] # Top left corner of ID 3
                if id == 4:
                    roi_corners["B"] = [int(corners[i][0][1][0]), int(corners[i][0][1][1])] # Top right corner of ID 4
            
            if 1 in ids and 2 in ids and 3 in ids and 4 in ids:

                # Here, I have used L2 norm. You can use L1 also.
                width_AD = np.sqrt(((roi_corners["A"][0] - roi_corners["D"][0]) ** 2) + ((roi_corners["A"][1] - roi_corners["D"][1]) ** 2))
                width_BC = np.sqrt(((roi_corners["B"][0] - roi_corners["C"][0]) ** 2) + ((roi_corners["B"][1] - roi_corners["C"][1]) ** 2))
                maxWidth = max(int(width_AD), int(width_BC))
                # maxWidth = self.new_width
                
                height_AB = np.sqrt(((roi_corners["A"][0] - roi_corners["B"][0]) ** 2) + ((roi_corners["A"][1] - roi_corners["B"][1]) ** 2))
                height_CD = np.sqrt(((roi_corners["C"][0] - roi_corners["D"][0]) ** 2) + ((roi_corners["C"][1] - roi_corners["D"][1]) ** 2))
                maxHeight = max(int(height_AB), int(height_CD))
                # maxHeight = self.new_height
                
                input_pts = np.float32([roi_corners["A"], roi_corners["B"], roi_corners["C"], roi_corners["D"]])
                output_pts = np.float32([[0, 0],
                                        [0, maxHeight - 1],
                                        [maxWidth - 1, maxHeight - 1],
                                        [maxWidth - 1, 0]])
                
                # Compute the perspective transform M
                M = cv2.getPerspectiveTransform(input_pts,output_pts)
                frame = cv2.warpPerspective(frame,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)
            
            else:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to RGB format

        aspect_ratio = frame.shape[1] / frame.shape[0]
        new_height = int(self.canvas_max_height)
        new_width = int(new_height * aspect_ratio)
        if new_width > self.canvas_max_width:
            new_width = self.canvas_max_width
            new_height = int(new_width / aspect_ratio)

        self.new_width = new_width
        self.new_height = new_height
    
        # Adjust the canvas size to match the image size
        self.canvas.config(width=new_width, height=new_height)

        frame = cv2.resize(frame, (new_width, new_height))
        image = Image.fromarray(frame)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.itemconfig(self.canvas_image, image=self.photo)
        self.canvas.config(scrollregion=self.canvas.bbox(self.canvas_image))  # adjust the scroll region to the image size

        # Repeat after an interval to capture continuously
        self.canvas.after(330, self.show_rectified_camera)

    def show_camera(self):

        if self.selected_camera != int(self.camera_dropdown.get()):
            self.selected_camera = int(self.camera_dropdown.get())
            self.cap.release()

            # Start a new video capture with the selected camera
            self.cap = cv2.VideoCapture(int(self.camera_dropdown.get()))

            # Get the width and height of the original image
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            # Calculate the scale factor to keep the aspect ratio and limit the height to 300
            scale = min(1, 300 / height)
            
            # Resize the original image
            self.resize_width = int(width * scale)
            self.resize_height = int(height * scale)


        # Get the latest frame and convert into Image
        frame = self.cap.read()[1]
        # Detect markers in the frame
        corners, ids, _ = cv2.aruco.detectMarkers(frame, self.dictionary)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        cv2image= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        cv2image_resized = cv2.resize(cv2image, (self.resize_width, self.resize_height))
        
        img = Image.fromarray(cv2image_resized)
        # Convert image to PhotoImage
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        
        # Repeat after an interval to capture continuously
        self.video_label.after(10, self.show_camera)

    def start_ocr(self):
        
        # Build ROI list
        roi_list = [{'variable': var.get(), 'ROI': rectangle} for var, rectangle in zip(self.rect_entries, self.rectangles) if rectangle is not None]
        process_webcam_feed(roi_list, self.selected_camera, self.new_width, self.new_height)
   

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
