import cv2
from pytesseract import pytesseract


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)



pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

# Load image
original_img = cv2.imread('images/cvt_spectrometer.jpg')
original_img = ResizeWithAspectRatio(original_img, height= 720)

# Apply contrast enhancement
img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
img = cv2.equalizeHist(img)

# Apply noise reduction
# img = cv2.medianBlur(img, 3)

# Apply adaptive thresholding
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)

# Define ROI
rois = cv2.selectROIs("Select ROIs", original_img)
cv2.destroyWindow("Select ROIs")
print(rois)

# Apply OCR
for roi in rois:
    roi_img = img[roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3]]
    roi_text = pytesseract.image_to_string(roi_img, config='--psm 11')

    if roi_text == '':
        roi_text = 'No text'

    # Draw bounding boxes around recognized text
    x = roi[0]
    y = roi[1]
    w = roi[2]
    h = roi[3]
    cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(original_img, roi_text, (x, y - 16), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)


# Display image
winname = 'Original image'
cv2.namedWindow(winname)
cv2.moveWindow(winname, 40,30)
cv2.imshow(winname, original_img)
cv2.waitKey(0)

img = ResizeWithAspectRatio(img, height= 720)
winname = 'Processed image'
cv2.namedWindow(winname)
cv2.moveWindow(winname, 40,30)
cv2.imshow(winname, img)
cv2.waitKey(0)

# gray = ResizeWithAspectRatio(gray, height= 720)
# winname = 'Gray'
# cv2.namedWindow(winname)
# cv2.moveWindow(winname, 40,30)
# cv2.imshow(winname, gray)
# cv2.waitKey(0)

# thresh = ResizeWithAspectRatio(thresh, height= 720)
# winname = 'Thresh'
# cv2.namedWindow(winname)
# cv2.moveWindow(winname, 40,30)
# cv2.imshow(winname, thresh)
# cv2.waitKey(0)

