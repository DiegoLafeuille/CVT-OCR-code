import csv
import datetime
import cv2


def do_roi_ocr(frame, reader, roi_list, cols):

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
        roi_img = img_processing_pipeline(roi_img)
        
        if roi['only_nums']:
            text = reader.readtext(roi_img, allowlist = '0123456789-+.')
        else:
            text = reader.readtext(roi_img)
        texts.append(text[0][1] if text else "No text recognized")

    # Write the extracted text to the csv file
    with open('results.csv', mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=cols)
        writer.writerow({cols[i]: texts[i] for i in range(len(texts))})

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
    # img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)[1]
    otsu_thresh, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
