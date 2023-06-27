import cv2
import copy
import numpy as np

class Font:
    def __init__(self, name, proc_pipeline, ocr_engine):
        self.name = name
        self.proc_pipeline = proc_pipeline
        self.ocr_engine = ocr_engine

def no_processing_pipeline(image):
    '''Image returned without being processed.'''
    return image

def default_pipeline(image):
    '''Default image processing pipeline.'''

    img = copy.copy(image)

    # # Denoise image
    # img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

    # Grayscale with gray value equal to highest value between R, G and B
    img = np.max(img, axis=2)
    
    # # Blur to remove noise
    # img = cv2.GaussianBlur(img,(7,7),0)

    # # Threshold
    # img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # Turn back to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img

def ld_7_seg_pipeline(image):
    '''Image processing pipeline for seven segments displays with light font on dark background.'''
    
    # Tranform image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Blur to remove noise
    blur = cv2.GaussianBlur(gray,(7,7),0)

    # Initial threshholding
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Turn image back into RGB to have the right shape
    processed_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    return processed_img

def dl_7_seg_pipeline(image):
    '''Image processing pipeline for seven segments displays with dark font on light background.'''
    
    # Tranform image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Blur to remove noise
    blur = cv2.GaussianBlur(gray,(7,7),0)

    # Initial threshholding
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Turn image back into RGB to have the right shape
    processed_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    return processed_img


fonts = [
    Font("None", no_processing_pipeline, "easyocr"),
    Font("Default", default_pipeline, "easyocr"),
    Font("Light on dark, 7-segments", ld_7_seg_pipeline, "tesseract"),
    Font("Dark on light, 7-segments", dl_7_seg_pipeline, "tesseract")
]