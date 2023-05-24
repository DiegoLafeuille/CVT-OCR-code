import gxipy as gx
from PIL import Image
import numpy
import cv2


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



# device_manager = gx.DeviceManager()
# dev_num, dev_info_list = device_manager.update_device_list()

cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
# print(dev_num)
# print(dev_info_list)
# cam = device_manager.open_device_by_index(1)

while True:
    ret, frame = cap.read()
    # print(frame.shape[1], frame.shape[0])
    pimg = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    new_width, new_height = resize_with_ratio(720, 720, pimg.shape[1], pimg.shape[0])
    pimg = cv2.resize(pimg, (new_width, new_height), interpolation=cv2.INTER_AREA)
    cv2.imshow("Image",pimg)
    cv2.waitKey(10)

cv2.destroyAllWindows()  