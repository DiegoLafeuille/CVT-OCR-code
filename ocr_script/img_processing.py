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

    # Grayscale with gray value equal to highest value between R, G and B
    img = np.max(img, axis=2)

    # Turn back to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img

def normal_grayscale_pipeline(image):
    '''Default image processing pipeline.'''

    img = copy.copy(image)

    # Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Turn back to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img

def seven_seg_pipeline(image):

    img = copy.copy(image)
    img = np.max(img, axis=2)

    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    processed_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return processed_img



fonts = [
    Font("Default", default_pipeline, "easyocr"),
    Font("Norm. gray", normal_grayscale_pipeline, "easyocr"),
    Font("7-segments display", seven_seg_pipeline, "tesseract"),
    Font("None", no_processing_pipeline, "easyocr"),
]