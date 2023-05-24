# version:1.0.1905.9051
import gxipy as gx
from PIL import Image
import numpy
import cv2


def main():
    # print the demo information
    print("")
    print("-------------------------------------------------------------")
    print("Sample to show how to acquire color image continuously and show acquired image.")
    print("-------------------------------------------------------------")
    print("")
    print("Initializing......")
    print("")

    

    # create a device manager
    device_manager = gx.DeviceManager()
    dev_num, dev_info_list = device_manager.update_device_list()
    
    if dev_num is 0:
        print("Number of enumerated devices is 0")
        return

    # open the first device
    cam = device_manager.open_device_by_index(1)

    # exit when the camera is a mono camera
    if cam.PixelColorFilter.is_implemented() is False:
        print("This sample does not support mono camera.")
        cam.close_device()
        return

    # set continuous acquisition
    cam.TriggerMode.set(gx.GxSwitchEntry.OFF)

    # set exposure
    cam.ExposureTime.set(361961.0)

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
    

    # acquisition image: num is the image number
    num = 100
    for i in range(num):
        # get raw image
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

        # show acquired image
        #img = Image.fromarray(numpy_image, 'RGB')
	#img.show()

	#display image with opencv
        pimg = cv2.cvtColor(numpy.asarray(numpy_image),cv2.COLOR_BGR2RGB)
	#cv2.imwrite("cat2.jpg", pimg)

        new_width, new_height = resize_with_ratio(720, 720, pimg.shape[1], pimg.shape[0])
        pimg = cv2.resize(pimg, (new_width, new_height), interpolation=cv2.INTER_AREA)
        cv2.imshow("Image",pimg)
        cv2.waitKey(10)
	

        # print height, width, and frame ID of the acquisition image
        print("Frame ID: %d   Height: %d   Width: %d, counter: %d"
              % (raw_image.get_frame_id(), raw_image.get_height(), raw_image.get_width(), i))

    # stop data acquisition
    cam.stream_off()

    # close device
    cam.close_device()
    cv2.destroyAllWindows()  

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

if __name__ == "__main__":
    main()
