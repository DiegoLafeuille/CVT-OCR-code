import cv2

class Font:
    def __init__(self, name, proc_pipeline, ocr_engine):
        self.name = name
        self.proc_pipeline = proc_pipeline
        self.ocr_engine = ocr_engine

def default_pipeline(image):
    '''Default image processing pipeline doesn't do anything to the image.'''
    
    return image

def br_pipeline(image):
    '''Image processing pipeline for black font on red background.'''

    # Tranform image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Blur to remove noise
    blur = cv2.GaussianBlur(gray,(7,7),0)

    # Initial threshholding
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Distance transform and threshholding of distance map
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    normed_dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    normed_dist = (normed_dist * 255).astype("uint8")
    _, thresh = cv2.threshold(normed_dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # "Opening" morphological operation to disconnect components
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    processed_img = cv2.dilate(thresh,kernel,iterations = 3)

    return processed_img

def ld_7_seg_pipeline(image):
    '''Image processing pipeline for seven segments displays with light font on dark background.'''
    
    # Tranform image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Blur to remove noise
    blur = cv2.GaussianBlur(gray,(7,7),0)

    # Initial threshholding
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

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
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Turn image back into RGB to have the right shape
    processed_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    return processed_img


fonts = [
    Font("Default", default_pipeline, "easyocr"),
    Font("Black on red", br_pipeline, "easyocr"),
    Font("Light on dark, 7-segments", ld_7_seg_pipeline, "tesseract"),
    Font("Dark on light, 7-segments", dl_7_seg_pipeline, "tesseract")
]