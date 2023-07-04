import gxipy as gx
from PIL import Image
import numpy
import cv2


# Set lens name
lens = "daheng_12mm"

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
cam.ExposureTime.set(150000.0)

# set auto white balance
cam.BalanceWhiteAuto.set(1)

# set gain
cam.Gain.set(10.0)

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

# Initialize a variable to store the image
img_counter = 0

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)


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
    # resized_frame = cv2.resize(frame,(int(width/5), int(height/5)), interpolation=cv2.INTER_AREA)
    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)
    
    # Check if the user pressed the space bar
    if key == ord(' '):
        # Increment the image counter
        img_counter += 1
        
        # Save the image to disk
        filename = f"image_{img_counter}.png"
        cv2.imwrite("cam_calibration/cameras/" + lens + "/" + filename, frame)
        
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