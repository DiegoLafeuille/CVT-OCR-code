import csv
import time
import cv2
import easyocr
import datetime


def process_webcam_feed(roi_list):
    # Create the csv file and write the headers
    with open('results.csv', mode='w', newline='') as file:
        fieldnames = ['Timestamp'] + [roi['variable'] for roi in roi_list]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

    # Initialize the video capture device and the OCR reader
    cap = cv2.VideoCapture(0)
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
            roi_img = frame[y1:y2, x1:x2]
            text = reader.readtext(roi_img)
            texts.append(text[0][1] if text else "")

        # Write the extracted text to the csv file
        with open('results.csv', mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow({fieldnames[i]: texts[i] for i in range(len(texts))})

        # Wait for 5 seconds before capturing the next frame
        time.sleep(5)

    # Release the video capture device and close the csv file
    cap.release()
