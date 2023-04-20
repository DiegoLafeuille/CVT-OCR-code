import csv
import time
import cv2
import easyocr
import datetime
import numpy as np


def process_webcam_feed(roi_list, selected_cam, canvas_width, canvas_height):
    # Create the csv file and write the headers
    with open('results.csv', mode='w', newline='') as file:
        fieldnames = ['Timestamp'] + [roi['variable'] for roi in roi_list]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

    # Initialize the video capture device and the OCR reader
    cap = cv2.VideoCapture(selected_cam)
    reader = easyocr.Reader(['en'], gpu=False)

    # Continuously capture frames from the webcam and extract text from the ROIs
    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            break

        # Extract text from the ROIs using easyOCR
        texts = []
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        texts.append(timestamp)
        for roi in roi_list:
            x1, y1, x2, y2 = roi['ROI']
            
            # Correction for scale of canvas rectangle
            scaling_factor_x = frame.shape[1] / canvas_width
            scaling_factor_y = frame.shape[0] / canvas_height
            x1 = int(x1 * scaling_factor_x)
            y1 = int(y1 * scaling_factor_y)
            x2 = int(x2 * scaling_factor_x)
            y2 = int(y2 * scaling_factor_y)
            roi_img = frame[y1:y2, x1:x2]

            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            roi_img = frame[y1:y2, x1:x2]
            text = reader.readtext(roi_img)
            texts.append(text[0][1] if text else "")
        
        # cv2.imshow('frame', frame)
        # cv2.waitKey(1)

        # Write the extracted text to the csv file
        with open('results.csv', mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow({fieldnames[i]: texts[i] for i in range(len(texts))})

        # Wait for 5 seconds before capturing the next frame
        time.sleep(5)

    # Release the video capture device and close the csv file
    cap.release()

