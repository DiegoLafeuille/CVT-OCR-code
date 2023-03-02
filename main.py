from PIL import Image
from pytesseract import pytesseract


# # If you don't have tesseract executable in your PATH, include the following:
pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
# # Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'


# # Simple image to string
print(pytesseract.image_to_string(Image.open('images/cvt_spectrometer.jpg'), config='--psm 11'))

# # In order to bypass the image conversions of pytesseract, just use relative or absolute image path
# # NOTE: In this case you should provide tesseract supported images or tesseract will return error
# print(pytesseract.image_to_string('images/cvt_spectrometer.jpg'))

# # List of available languages
# print(pytesseract.get_languages(config=''))

# # French text image to string
# print(pytesseract.image_to_string(Image.open('test-european.jpg'), lang='fra'))

# # Batch processing with a single file containing the list of multiple image file paths
# print(pytesseract.image_to_string('images.txt'))

# # Timeout/terminate the tesseract job after a period of time
# try:
#     print(pytesseract.image_to_string('test.jpg', timeout=2)) # Timeout after 2 seconds
#     print(pytesseract.image_to_string('test.jpg', timeout=0.5)) # Timeout after half a second
# except RuntimeError as timeout_error:
#     # Tesseract processing is terminated
#     pass

# # Get bounding box estimates
# print(pytesseract.image_to_boxes(Image.open('images/cvt_spectrometer.jpg')))

# # Get verbose data including boxes, confidences, line and page numbers
# print(pytesseract.image_to_data(Image.open('images/cvt_spectrometer.jpg')))

# # Get information about orientation and script detection
# print(pytesseract.image_to_osd(Image.open('images/cvt_spectrometer.jpg')))

# # Get a searchable PDF
# pdf = pytesseract.image_to_pdf_or_hocr('images/cvt_spectrometer.jpg', extension='pdf')
# with open('test.pdf', 'w+b') as f:
#     f.write(pdf) # pdf type is bytes by default

# # Get HOCR output
# hocr = pytesseract.image_to_pdf_or_hocr('images/cvt_spectrometer.jpg', extension='hocr')

# # Get ALTO XML output
# xml = pytesseract.image_to_alto_xml('images/cvt_spectrometer.jpg')



# import cv2


# img_cv = cv2.imread(r'images/cvt_spectrometer.jpg')

# # By default OpenCV stores images in BGR format and since pytesseract assumes RGB format,
# # we need to convert from BGR to RGB format/mode:
# # img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
# # print(pytesseract.image_to_string(img_rgb))
# # OR
# # img_rgb = Image.frombytes('RGB', img_cv.shape[:2], img_cv, 'raw', 'BGR', 0, 0)
# print(pytesseract.image_to_string(img_cv))