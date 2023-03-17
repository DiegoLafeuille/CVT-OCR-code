import cv2
from pytesseract import pytesseract, Output
import numpy as np

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

# Load image
original_img = cv2.imread('images/cvt_spectrometer_one_var.jpg')

# Apply contrast enhancement
img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
# img = cv2.equalizeHist(img)

# Apply noise reduction
img = cv2.medianBlur(img, 3)

# # Apply morphological opening to remove noise
# kernel = np.ones((3,3), np.uint8)
# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)

# # Apply Hough transform to detect lines
# lines = cv2.HoughLines(opening, 1, np.pi/180, 200)

# # Calculate angle of rotation needed to align lines
# angle = 0
# if lines is not None:
#     for line in lines:
#         rho, theta = line[0]
#         if np.degrees(theta) < 5 or np.degrees(theta) > 175:
#             continue
#         angle += np.degrees(theta)
#     angle /= len(lines)
#     angle -= 90

# # Rotate image to correct skew
# h, w = img.shape[:2]
# center = (w // 2, h // 2)
# M = cv2.getRotationMatrix2D(center, angle, 1.0)
# rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# Apply adaptive thresholding
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21,6)

# Apply OCR
pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
image_data = pytesseract.image_to_data(img, output_type=Output.DICT, config='--psm 11')

# Draw bounding boxes around recognized text
for i, word in enumerate(image_data['text']):
    if word != '':
        x, y, w, h = image_data['left'][i], image_data['top'][i], image_data['width'][i], image_data['height'][i]
        cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(original_img, word, (x, y - 16), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

# Display image
original_img = ResizeWithAspectRatio(original_img, height= 720)
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

# print(image_data['text'])