import gxipy as gx
import numpy
import cv2
import os
import re
import copy
import argparse


# Argument parsing
parser = argparse.ArgumentParser(description='Capture images from a Daheng camera.')
parser.add_argument('--camera', '-c', required= True, help='Name of the Danheng camera (camera/lens combination)')
parser.add_argument('--exposure', '-x', default='100', help='Exposure time (1-999)[ms]')
parser.add_argument('--gain', '-g', default='10', help='Analog gain (0.0-24.0)[dB]')
args = parser.parse_args()

# Set arguments
camera = args.camera
exposure = args.exposure * 1000
gain = args.gain

# Set lens name
lens = "daheng_25mm"

# Setup Daheng camera

# create a device manager
device_manager = gx.DeviceManager()
dev_num, dev_info_list = device_manager.update_device_list()

if dev_num == 0:
    print("Number of enumerated devices is 0")

# open the first device
cam = device_manager.open_device_by_index(1)

# exit when the camera is a mono camera
if cam.PixelColorFilter.is_implemented() is False:
    print("This sample does not support mono camera.")
    cam.close_device()

# set continuous acquisition
cam.TriggerMode.set(gx.GxSwitchEntry.OFF)

# set exposure
cam.ExposureTime.set(exposure)

# # set auto white balance
# cam.BalanceWhiteAuto.set(1)

# set gain
cam.Gain.set(gain)

# get param of improving image quality
if cam.GammaParam.is_readable():
    gamma_value = cam.GammaParam.get()
    gamma_lut = gx.Utility.get_gamma_lut(gamma_value)
else:
    gamma_lut = None
if cam.ContrastParam.is_readable():
    contrast_value = cam.ContrastParam.get()
    contrast_lut = gx.Utility.get_contrast_lut(contrast_value)
else:
    contrast_lut = None
if cam.ColorCorrectionParam.is_readable():
    color_correction_param = cam.ColorCorrectionParam.get()
else:
    color_correction_param = 0


# set the acq buffer count
cam.data_stream[0].set_acquisition_buffer_number(1)

# start data acquisition
cam.stream_on()

raw_image = cam.data_stream[0].get_image()
if raw_image is None:
    print("Getting image failed.")
numpy_image = raw_image.get_numpy_array()
height, width = numpy_image.shape
print(f"{width}x{height}")


# Specify the folder path where the images will be saved
folder_path = "cam_calibration/cameras/" + lens + "/"

# Check if the folder exists, create it if necessary
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Get a list of existing image files in the folder
existing_images = [f for f in os.listdir(folder_path) if f.endswith(".png")]

# Initialize the counter with the highest number from existing images
img_counter = 0
if existing_images:
    # Extract the image number from the filename and find the maximum
    img_numbers = [int(re.search(r"image_(\d+)\.png", f).group(1)) for f in existing_images if re.search(r"image_(\d+)\.png", f)]
    img_counter = max(img_numbers)


while True:
    # Read a frame from the camera
    raw_image = cam.data_stream[0].get_image()
    if raw_image is None:
        print("Getting image failed.")
        continue

    # get RGB image from raw image
    rgb_image = raw_image.convert("RGB")
    if rgb_image is None:
        continue

    # improve image quality
    rgb_image.image_improvement(color_correction_param, contrast_lut, gamma_lut)

    # create numpy array with data from raw image
    numpy_image = rgb_image.get_numpy_array()
    if numpy_image is None:
        continue
    
    frame = cv2.cvtColor(numpy.asarray(numpy_image),cv2.COLOR_BGR2RGB)
    
    # Show the frame in a window
    resized_frame = copy.copy(frame)
    resized_frame = cv2.resize(frame,(int(width/5), int(height/5)), interpolation=cv2.INTER_AREA)
    cv2.imshow("Camera", resized_frame)

    key = cv2.waitKey(1)
    
    # Check if the user pressed the space bar
    if key == ord(' '):
        # Increment the image counter
        img_counter += 1
        
        # Save the image to disk
        filename = f"image_{img_counter}.png"
        cv2.imwrite(f"{folder_path}{filename}", frame)
        
        # Print a message to the console
        print(f"{filename} saved!")
        
    # Check if the user pressed the escape key
    elif key == 27:
        break

# stop data acquisition
cam.stream_off()

# close device
cam.close_device()
cv2.destroyAllWindows()  