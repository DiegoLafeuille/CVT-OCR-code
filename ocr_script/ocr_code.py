import csv
import time
import cv2
import easyocr
import datetime
import numpy as np


def process_webcam_feed(frame, reader, roi_list, cols):

    # print(f"OCR: resolution: {frame.shape[1]}x{frame.shape[0]}")
    
    # Extract text from the ROIs using easyOCR
    texts = []
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    texts.append(timestamp)
    for roi in roi_list:

        x1 = min(roi['ROI'][0],roi['ROI'][2])
        x2 = max(roi['ROI'][0],roi['ROI'][2])
        y1 = min(roi['ROI'][1],roi['ROI'][3])
        y2 = max(roi['ROI'][1],roi['ROI'][3])
        roi_img = frame[y1:y2, x1:x2]
        
        if roi['only_nums']:
            text = reader.readtext(roi_img, allowlist = "0123456789.,'")
        else:
            text = reader.readtext(roi_img)
        texts.append(text[0][1] if text else "No text recognized")

    suffix = 0
    # Write the extracted text to the csv file
    with open('results.csv', mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=cols)
        writer.writerow({cols[i]: texts[i] for i in range(len(texts))})

