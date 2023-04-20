import csv
import time
import cv2
import easyocr
import datetime
import numpy as np


def process_webcam_feed(frame, reader, roi_list, cols):

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
        writer = csv.DictWriter(file, fieldnames=cols)
        writer.writerow({cols[i]: texts[i] for i in range(len(texts))})

