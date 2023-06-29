import os
import csv
import datetime
import easyocr
import pytesseract



def initialize_ocr_engines(ocr_engines):

    if "easyocr" in ocr_engines:
        global easyocr_reader
        easyocr_reader = easyocr.Reader(['en'], gpu=False)
    
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

        # Process image with appropriate processing pipeline
        img_processing_pipeline =  roi['font'].proc_pipeline
        roi_img = img_processing_pipeline(roi_img)

        # Call ocr_function with appropriate engine
        if roi['font'].ocr_engine == "easyocr":
            text = easyocr_ocr(roi_img, roi['only_nums'])
        elif roi['font'].ocr_engine == "tesseract":
            text = tesseract_ocr(roi_img, roi['only_nums'])
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
        texts = easyocr_reader.readtext(
            img, 
            detail = 0, 
        )
        
    text = ''.join(texts)
    return text if text else "No text recognized"
    
def tesseract_ocr(img, only_nums):
    if only_nums:
        text = pytesseract.image_to_string(img, lang="lets", config="--psm 7 -c tessedit_char_whitelist=+-,.0123456789")
    else:
        text = pytesseract.image_to_string(img, lang="lets", config="--psm 7")
    return text if text else "No text recognized"
