import easyocr
import cv2
import torch
import os

if not torch.cuda.is_available():
    print("No CUDA found")
    exit()

print(torch.cuda.get_device_name(0))

# Create a reader to do OCR.
# gpu must be True to use GPU (only if your system supports it)
reader = easyocr.Reader(['en'], gpu=True)

# Read your images
images = list(os.walk("experiment/slides_walkiria"))[0][2]

for img in images:

    image_path = "experiment/slides_walkiria/" + img
    image = cv2.imread(image_path)

    # Do OCR on the image
    result = reader.readtext(image)

    # print the text
    for detection in result:
        text = detection[1]
        print("Image: ", img[:3])
        print("Detected Text: ", text)
        print("-----------")