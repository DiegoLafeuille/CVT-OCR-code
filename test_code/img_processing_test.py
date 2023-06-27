import cv2
import easyocr
import copy
import numpy as np
import time
import math


def no_pipeline(image):
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

def br_pipeline(image):
    '''Image processing pipeline for black font on red background.'''

    img = image

    # Denoise image
    img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

    # Tranform image to grayscale
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Blur to remove noise
    img = cv2.GaussianBlur(img,(7,7),0)

    # # Enhance contrast
    img = cv2.equalizeHist(img)
    # # clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(5,5))
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # img = clahe.apply(img)

    # img = cv2.fastNlMeansDenoising(img,None,10,21,7)

    # Initial threshholding
    # img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    # img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    img = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)[1]
    # img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # # Distance transform and threshholding of distance map
    # img = cv2.distanceTransform(img, cv2.DIST_L2, 3)
    # img = cv2.normalize(img, img, 0, 1.0, cv2.NORM_MINMAX)
    # img = (img * 255).astype("uint8")
    # img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

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

def resize_with_ratio(max_width, max_height, width, height):

    # Calculate the aspect ratio of the original image
    aspect_ratio = width / float(height)

    # Calculate the maximum aspect ratio allowed based on the given maximum width and height
    max_aspect_ratio = max_width / float(max_height)

    # If the original aspect ratio is greater than the maximum allowed aspect ratio,
    # then the width should be resized to the maximum width, and the height should be
    # resized accordingly to maintain the aspect ratio.
    if aspect_ratio > max_aspect_ratio:
        resized_width = int(max_width)
        resized_height = int(max_width / aspect_ratio)
    # Otherwise, the height should be resized to the maximum height, and the width should
    # be resized accordingly to maintain the aspect ratio.
    else:
        resized_width = int(max_height * aspect_ratio)
        resized_height = int(max_height)

    # Return the resized width and height as a tuple
    return resized_width, resized_height



def main():

    reader = easyocr.Reader(['en'], gpu=False)

    # Open the video file
    video = cv2.VideoCapture('roi_video.avi')
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

    row_number = 4

    while video.isOpened():
        # Read a frame from the video
        ret, frame = video.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if not ret:
            # End of video
            break
        
        # Divide the frame into four rows
        height, width = frame.shape[:2]
        row_height = math.ceil(height / row_number)
        rows = [frame[i * row_height : (i + 1) * row_height, :] for i in range(row_number)]

        # process_time = 0
        # ocr_time = 0
        processed_rows = []
        texts = []

        # Process each row separately
        for i, row in enumerate(rows):
            # Apply image processing to the row of the frame
            # current_time = time.time()
            processed_row = default_pipeline(row)
            processed_rows.append(processed_row)
            # process_time =+ time.time() - current_time

            # current_time = time.time()
            
            text = reader.readtext(
                processed_row, 
                allowlist = '0123456789-+.,', 
                link_threshold=0.99, 
                detail = 0, 
                width_ths = 0.99,
                height_ths = 0.99,
            )
            texts.append(text)
            
            # ocr_time =+ time.time() - current_time
            # print([text[1] for text in texts])

        print(texts)

        # print(f"Image processing time: {process_time}")
        # print(f"OCR time: {ocr_time}")

        # Display the processed frame
        processed_frame = cv2.vconcat(processed_rows)
        frame = cv2.hconcat([frame, processed_frame])
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        height, width = frame.shape[:2]
        new_w, new_h = resize_with_ratio(1024, 720, width, height)
        frame = cv2.resize(frame, (new_w, new_h))

        cv2.imshow('Processed Video', frame)

        # Check for 'Escape' key press to exit
        if cv2.waitKey(33) == 27:
            break

    # Release the video file and close windows
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()