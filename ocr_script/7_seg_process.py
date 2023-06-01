import cv2
import easyocr
import copy
import numpy as np

def img_processing_pipeline(image):


    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # red_channel = img[:, :, 2]
    # pixels_to_modify = red_channel > 127
    # img[pixels_to_modify, 1] = 0
    # img[pixels_to_modify, 0] = 0

    # img[:,:,0] = 0 
    # img[:,:,1] = 0

    # Tranform image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Blur to remove noise
    img = cv2.GaussianBlur(img,(7,7),0)

    # Initial threshholding
    img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)[1]
    # otsu_thresh, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # print(otsu_thresh)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 1)

    # Distance transform and threshholding of distance map
    img = cv2.distanceTransform(img, cv2.DIST_L2, 5)
    img = cv2.normalize(img, img, 0, 1.0, cv2.NORM_MINMAX)
    img = (img * 255).astype("uint8")
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Get skeleton
    img = cv2.ximgproc.thinning(img)

    # "Opening" morphological operation to disconnect components
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img = cv2.dilate(img,kernel,iterations = 3)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


    return img




reader = easyocr.Reader(['en'], gpu=False)

# Open the video file
video = cv2.VideoCapture('roi_video.avi')
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames: {total_frames}")

while video.isOpened():
    # Read a frame from the video
    ret, frame = video.read()

    if not ret:
        # End of video
        break

    # Apply image processing to the frame
    # frame = img_processing_pipeline(frame)

    
    texts = reader.readtext(frame, allowlist = '0123456789-+.')
    print([text[1] for text in texts])

    # Display the processed frame
    cv2.imshow('Processed Video', frame)

    # Check for 'Escape' key press to exit
    if cv2.waitKey(1) == 27:
        break

# Release the video file and close windows
video.release()
cv2.destroyAllWindows()