import tkinter as tk
import cv2
import numpy as np
import easyocr
from PIL import Image, ImageTk
from tkinter_webcam import webcam

class OCR_GUI:

    def __init__(self, master):
        self.master = master
        self.master.title("OCR GUI")

        self.image = cv2.imread("images/cvt_spectrometer_zoomed.jpg")
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        

        canvas_max_width = 750
        canvas_max_height = 650

        # get and resize the image while maintaining its aspect ratio
        self.image = cv2.imread("images/cvt_spectrometer_zoomed.jpg")
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        aspect_ratio = self.image.shape[1] / self.image.shape[0]
        new_height = int(canvas_max_height)
        new_width = int(new_height * aspect_ratio)
        if new_width > canvas_max_width:
            new_width = canvas_max_width
            new_height = int(new_width / aspect_ratio)
        self.image = cv2.resize(self.image, (new_width, new_height))
        self.photo = self.get_image()

        self.left_frame = tk.Frame(master, bg='grey', width=canvas_max_width)
        self.left_frame.pack(side=tk.LEFT, fill='both', padx=15, pady=15)

        self.canvas = tk.Canvas(self.left_frame, bd=2, bg="grey", width=new_width, height=new_height)
        self.canvas.pack(side=tk.TOP, padx=15, pady=15)

        self.canvas.create_image(new_width/2, new_height/2, anchor=tk.CENTER, image=self.photo)
        
        self.refresh_button = tk.Button(self.left_frame, text="Refresh", command=self.refresh_image)
        self.refresh_button.pack(side=tk.BOTTOM, anchor='s', padx=15)

        self.rectangles = []
        self.rectangles_drawing = []
        self.rect_drawing_labels = []
        self.labels = []  # initialize the labels list

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.right_frame = tk.Frame(master, bg='grey', width=300)
        self.right_frame.pack(side=tk.RIGHT, fill='both', padx=15, pady=15)

        self.video=webcam.Box(self.right_frame, width=300, height=300)
        self.video.show_frames()

        self.list_frame = tk.Frame(self.right_frame, bg='white', width=300, height=300)
        self.list_frame.pack(side=tk.TOP, fill='x', padx=15, pady=15)

        self.button = tk.Button(self.right_frame, text="Read Text", command=self.read_text)
        self.button.pack(side=tk.BOTTOM, anchor='s', padx=15, pady=15)

        self.rect_labels = []
        self.rect_entries = []
        self.rect_delete = []

        self.results = []

    def get_image(self):
        image = Image.fromarray(self.image)
        image = ImageTk.PhotoImage(image)
        return image

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
        self.rect_labels[-1].grid(row=rect_number-1,column=0)
        self.rect_entries[-1].grid(row=rect_number-1,column=1)
        self.rect_delete[-1].grid(row=rect_number-1,column=2)

    def read_text(self):
        # use EasyOCR to read the text within the selected rectangles
        reader = easyocr.Reader(['en'])
        for i, rect in enumerate(self.rectangles):
            if rect == None: continue
            
            x1, y1, x2, y2 = rect
            # crop the image to the rectangle
            cropped_image = self.image[y1:y2, x1:x2]
            # convert the image to grayscale
            gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)
            # read the text from the grayscale image
            result = {
                'Variable': self.rect_entries[i].get(), 
                'Text': reader.readtext(gray_image)[0][1]
                }
            print(result)

    def refresh_image(self):
        pass

root = tk.Tk()
root.geometry = ("1080x720")
root.config(bg="skyblue")
gui = OCR_GUI(root)
root.mainloop()