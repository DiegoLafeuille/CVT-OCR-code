import os
import csv
import datetime
import easyocr
import pytesseract
import numpy as np
import cv2



def initialize_ocr_engines(ocr_engines):

    if "easyocr" in ocr_engines:
        global easyocr_reader
        easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    
    if "tesseract" in ocr_engines:
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
        os.environ['TESSDATA_PREFIX'] = r".\ocr_script\Tesseract_sevenSegmentsLetsGoDigital\tessdata"
    
def ocr_on_roi(frame, roi_list, cols):

    # Extract text from the ROIs using easyOCR
    texts = []
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    texts.append(timestamp)

    for roi in roi_list:

        # Crop frame on ROI
        x1 = min(roi['ROI'][0],roi['ROI'][2])
        x2 = max(roi['ROI'][0],roi['ROI'][2])
        y1 = min(roi['ROI'][1],roi['ROI'][3])
        y2 = max(roi['ROI'][1],roi['ROI'][3])
        roi_img = frame[y1:y2, x1:x2]

        # Crop ROI on text
        cropped_img, _ = crop_roi(roi_img)

        # Process image with appropriate processing pipeline
        img_processing_pipeline =  roi['font'].proc_pipeline
        processed_roi = img_processing_pipeline(cropped_img)

        # Call ocr_function with appropriate engine
        if roi['font'].ocr_engine == "easyocr":
            text = easyocr_ocr(processed_roi, roi['only_nums'])
        elif roi['font'].ocr_engine == "tesseract":
            text = tesseract_ocr(processed_roi, roi['only_nums'])
        texts.append(text)
        
    # Write the extracted text to the csv file
    with open('results.csv', mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=cols, quoting=csv.QUOTE_ALL)
        writer.writerow({cols[i]: texts[i] for i in range(len(texts))})

def easyocr_ocr(img, only_nums):
    
    if only_nums:
        
        # texts = easyocr_reader.readtext(
        #     img, 
        #     allowlist = '0123456789-+.', 
        #     link_threshold=0.99, 
        #     detail = 0, 
        #     width_ths = 0.99,
        #     height_ths = 0.99,
        # )
        texts = easyocr_reader.recognize(
            img, 
            batch_size = 5,
            allowlist = '0123456789-+.', 
            detail = 0, 
        )
    
    else:
        
        # texts = easyocr_reader.readtext(
        #     img, 
        #     link_threshold=0.99, 
        #     detail = 0, 
        #     width_ths = 0.99,
        #     height_ths = 0.99,
        # )
        texts = easyocr_reader.recognize(
            img, 
            batch_size = 5,
            detail = 0, 
            # contrast_ths = 0.4,
        )
        
    text = ''.join(texts)
    return text if text else "No text recognized"
    
def tesseract_ocr(img, only_nums):
    if only_nums:
        text = pytesseract.image_to_string(img, lang="lets", config="--psm 7 -c tessedit_char_whitelist=+-,.0123456789")
    else:
        text = pytesseract.image_to_string(img, lang="lets", config="--psm 7")
    return text if text else "No text recognized"

def crop_roi(img):
    
    # Convert the image to grayscale
    gray = np.max(img, axis=2)

    # Apply thresholding to convert the image to binary, characters need to be white, background black
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Invert image if text is black on white background (more white pixels overall)
    mean_intensity = np.mean(binary)
    if mean_intensity > 127:
        binary = 255 - binary
    
    # Apply morphological opening to remove spots due to noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find contours in the opened binary image
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Combine all contours into a single contour
    combined_contour = np.vstack(contours)

    # Get the bounding rectangle of the combined contour
    x, y, w, h = cv2.boundingRect(combined_contour)

    # Add a margin if not exceeding original size
    margin = 5
    x = max(x - margin, 0)
    y = max(y - margin, 0)
    w = min(w + 2 * margin, img.shape[1])
    h = min(h + 2 * margin, img.shape[0])

    # Draw the bounding box on the image
    image_with_box = np.copy(img)
    cv2.rectangle(image_with_box, (x, y), (x + w, y + h), (0, 0, 255), 6)
    # Draw contours on the image
    cv2.drawContours(image_with_box, contours, -1, (0, 0, 0), 3)
    
    # Crop the image using the bounding rectangle
    cropped_image = img[y:y+h, x:x+w]

    return cropped_image, image_with_box