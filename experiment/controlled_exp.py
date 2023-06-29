import os
import random
import numpy as np
import pickle
import gxipy as gx
import cv2
import img_processing as imgp
import easyocr
import pytesseract
import datetime
import json
from tqdm import tqdm


# Names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}


def import_params(filename):
    '''Imports all necessary data to be able to run script to do the 
    OCR without need of prior configuration through the GUI'''

    # Path to pickle file
    param_file_path = "ocr_script/automated_script_params/" + filename + ".pickle"

    # Load the GUI parameters from the pickle file
    with open(param_file_path, 'rb') as f:
        params = pickle.load(f)

    return params

def update_calibration(calib_file):

    print(f"Updating calibration to {calib_file}")

    # Path to pickle file
    calib_file_path = "cam_calibration/cameras/" + calib_file + "/calibration_params.pickle"

    # Load the calibration parameters from the pickle file
    with open(calib_file_path, 'rb') as f:
        calibration_params = pickle.load(f)

    # Extract the parameters from the dictionary
    mtx = calibration_params["mtx"]
    dist = calibration_params["dist"]
    calib_w = int(calibration_params["calib_w"])
    calib_h = int(calibration_params["calib_h"])

    if any(x is None for x in (mtx, dist)):
        print("Error: Failed to retrieve calibration parameters.")
    
    return mtx, dist, calib_w, calib_h

def update_cam_input(cam_input, cam_type, calib_w, calib_h):
    
    print(f"Changing camera to input {cam_input}")

    if cam_type == 'daheng':

        global device_manager
        
        try:
            device_manager = gx.DeviceManager()
            cam = device_manager.open_device_by_sn(cam_input)
            
            # set continuous acquisition
            cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
            
            # set exposure time
            # cam.ExposureAuto.set(1)
            cam.ExposureTime.set(100000.0)
            
            # # set auto white balance
            # cam.BalanceWhiteAuto.set(1)

            cam.Gain.set(10.0)

            # get param of improving image quality
            color_correction_param, contrast_lut, gamma_lut = improve_daheng_image(cam)

            cam.data_stream[0].set_acquisition_buffer_number(1)
            cam.stream_on()

            raw_image = cam.data_stream[0].get_image()
            if raw_image is None:
                raise Exception("FrameNotRead")
            
            numpy_image = raw_image.get_numpy_array()
            
            height, width = numpy_image.shape
            if calib_w != width or calib_h != height:
                print("Warning", "Target and actual resolutions differ.\n"
                                    + "Make sure camera input and camera name correspond.\n"
                                    + f"Target resolution = {calib_w}x{calib_h}\n"
                                    + f"Actual resolution = {width}x{height}")
                
            return cam, color_correction_param, contrast_lut, gamma_lut
            

                                
        except Exception as e:
            if str(e) == "FrameNotRead":
                print("Error", "Unable to retrieve video feed from this camera.\n"
                                    + "Please check the connection and make sure the camera is not being used by another application.")
            else:
                print("Error", "An error occurred while trying to retrieve video feed from this camera.")
                raise e
                    
    else:

        try :
            if cam is not None:
                cam.release()
            cam = cv2.VideoCapture(int(cam_input))
            cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, calib_w)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, calib_h)
            ret,frame = get_frame(cam, cam_type)
            if not ret:
                raise Exception("FrameNotRead")
            width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

            if calib_w != int(width) or calib_h != int(height):
                print("Warning: Target and actual resolutions differ.\n"
                                    + "Make sure camera input and camera name correspond.\n"
                                    + f"Target resolution = {calib_w}x{calib_h}\n"
                                    + f"Actual resolution = {int(width)}x{int(height)}")

            return cam, None, None, None

        except Exception as e:
            if str(e) == "FrameNotRead":
                print("Error: Unable to retrieve video feed from this camera.\n"
                                    + "Please check the connection and make sure the camera is not being used by another application.")
            else:
                print("Error: An error occurred while trying to retrieve video feed from this camera.")
                raise e
        
def improve_daheng_image(cam):

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

    return color_correction_param, contrast_lut, gamma_lut

def get_frame(cam, cam_type, color_correction_param = None, contrast_lut = None, gamma_lut = None):

    if cam_type == 'daheng':
        ret = True

        # Seems to be a queue of three stored images, so for-loop to get past images of previous slides
        for i in range(10):
            # cam.data_stream[0].flush_queue()
            raw_image = cam.data_stream[0].get_image()

        rgb_image = raw_image.convert("RGB")
        rgb_image.image_improvement(color_correction_param, contrast_lut, gamma_lut)
        frame = rgb_image.get_numpy_array()
        if frame is None:
            ret = False

    else:
        ret, frame = cam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return ret, frame

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

def find_img_coords(world_coords, rvec, tvec, mtx, dist):
    
    point_3d = np.array(world_coords).reshape((1, 1, 3))

    # Use projectPoints to project the 3D point onto the 2D image plane
    point_2d, _ = cv2.projectPoints(point_3d, rvec, tvec, mtx, dist)

    # Extract the pixel coordinates of the projected point
    pixel_coords = tuple(map(int, point_2d[0, 0]))

    return pixel_coords

def get_surface_dims(surface_world_coords, calib_w, calib_h):

    width_AD = np.linalg.norm(surface_world_coords[0] - surface_world_coords[3])
    width_BC = np.linalg.norm(surface_world_coords[1] - surface_world_coords[2])
    surface_w = max(width_AD, width_BC)

    height_AB = np.linalg.norm(surface_world_coords[0] - surface_world_coords[1])
    height_CD = np.linalg.norm(surface_world_coords[2] - surface_world_coords[3])
    surface_h = max(height_AB, height_CD)


    # Resize for max image size within original size while keeping surface ratio
    new_width, new_height = resize_with_ratio(calib_w, calib_h, surface_w, surface_h)

    return new_width, new_height

def draw_crosshair(frame):
    
    # Draw crosshair
    crosshair_color = (0, 0, 255)
    crosshair_thickness = 2
    crosshair_length = 20

    h, w = frame.shape[:2]
    center_x = w // 2
    center_y = h // 2

    # Draw vertical line
    cv2.line(frame, (center_x, center_y - crosshair_length),
            (center_x, center_y + crosshair_length), crosshair_color, crosshair_thickness)

    # Draw horizontal line
    cv2.line(frame, (center_x - crosshair_length, center_y),
            (center_x + crosshair_length, center_y), crosshair_color, crosshair_thickness)
    
    return frame

def rectify_image(frame, params, calib_w, calib_h, mtx, dist):
    
    # Get video feed resolution
    # height, width = frame.shape[:2]
    # print(f"Rectified original frame resolution {width}x{height}")

    # Detect markers in the frame
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[params["Aruco dictionary"]])
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

    warp_input_pts = []

    board = cv2.aruco.CharucoBoard((params["Charuco width"], params["Charuco heigt"]), params["Square size"], params["Aruco size"], aruco_dict)
    cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedImgPoints)

    # If there are markers found by detector
    if not np.all(ids is not None):
        print("No markers found")
        return None
    
    charucoretval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
    frame = cv2.aruco.drawDetectedCornersCharuco(frame, charucoCorners, charucoIds, (0,255,0))
    retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, mtx, dist, np.zeros((3, 1)), np.zeros((3, 1)))  

    if not retval:
        print("No pose estimated")
        return None
        
    # Get surface image coordinates
    surface_world_coords = params["Target surface world coordinates"]
    surface_img_coords = [find_img_coords(point_coords, rvec, tvec, mtx, dist) for point_coords in surface_world_coords]

    #  Updating detected dimensions of object every 50 consecutive frames where surface is found
    new_width, new_height = get_surface_dims(surface_world_coords, calib_w, calib_h)
    
    for point in surface_img_coords:
        point_canvas_coords = [int(point[0]), int(point[1])]
        warp_input_pts.append(point_canvas_coords)
    warp_input_pts = np.float32(warp_input_pts)
    
    # If surface coordinates are found
    if len(warp_input_pts) > 0:
        
        # Compute the perspective transform M and warp frame
        warp_output_pts = np.float32([[0, 0],
                                [0, new_height - 1],
                                [new_width - 1, new_height - 1],
                                [new_width - 1, 0]])

        M = cv2.getPerspectiveTransform(warp_input_pts,warp_output_pts)

        rectified_frame = cv2.warpPerspective(frame,M,(new_width, new_height),flags=cv2.INTER_CUBIC)
    
    else:
        rectified_frame = None

    return rectified_frame

def get_roi(frame, code, roi_list):

    # Get correct ROI coordinates for the size of this slide's text
    if code[1] == "0":
        roi = next((rectangle['ROI'] for rectangle in roi_list if rectangle['variable'] == 'Big'), None)
    if code[1] == "1":
        roi = next((rectangle['ROI'] for rectangle in roi_list if rectangle['variable'] == 'Medium'), None)
    if code[1] == "2":
        roi = next((rectangle['ROI'] for rectangle in roi_list if rectangle['variable'] == 'Small'), None)
    
    if roi is None:
        print("ROI list from export is missing an ROI.")
        exit()

    # Crop frame on ROI
    x1 = min(roi[0],roi[2])
    x2 = max(roi[0],roi[2])
    y1 = min(roi[1],roi[3])
    y2 = max(roi[1],roi[3])
    roi_img = frame[y1:y2, x1:x2]

    return roi_img

def process_img(img, code):
    
    # Get correct processing pipeline for conventional fonts and different colors
    if code[0] != "4" and code[2] == "0":
        process_pipeline = imgp.default_pipeline
    elif code[0] != "4" and code[2] == "1":
        process_pipeline = imgp.default_pipeline
    elif code[0] != "4" and code[2] == "2":
        process_pipeline = imgp.default_pipeline
    elif code[0] != "4" and code[2] == "3":
        process_pipeline = imgp.default_pipeline

    # Get correct processing pipeline for Let's-Go-Digital (7 segments) font and different colors
    elif code[0] == "4" and code[2] == "0":
        process_pipeline = imgp.dl_7_seg_pipeline
    elif code[0] == "4" and code[2] == "1":
        process_pipeline = imgp.ld_7_seg_pipeline
    elif code[0] == "4" and code[2] == "2":
        process_pipeline = imgp.dl_7_seg_pipeline
    elif code[0] == "4" and code[2] == "3":
        process_pipeline = imgp.dl_7_seg_pipeline

    return process_pipeline(img)

def call_ocr(img, code, reader):
    
    # Check which OCR engine has to be used based on text font
    if code[0] != "4":
        # texts = reader.readtext(
        #     img, 
        #     allowlist = '0123456789-+.', 
        #     link_threshold=0.99, 
        #     detail = 0, 
        #     width_ths = 0.99,
        #     height_ths = 0.99,
        #     batch_size = 5,
        # )
        texts = reader.recognize(
            img, 
            allowlist = '0123456789-+.', 
            # link_threshold=0.99, 
            detail = 0, 
            # width_ths = 0.99,
            # height_ths = 0.99,
        )
        text = ''.join(texts)
    if code[0] == "4":
        text = pytesseract.image_to_string(img, lang="lets", config="--psm 7 -c tessedit_char_whitelist=+-.0123456789")

    return text

def close_cam(cam, cam_type):

    if cam_type == "daheng":
        cam.stream_off()
        cam.close_device()

    else:
        cam.release()



def main():

    # Start timer to time full experiment cycle duration
    start_time = datetime.datetime.now()

    # Debugging image display variables
    display_rois = True
    show_frame = False

    # Get GUI parameters for automated script
    filename = "controlled_experiment_params"
    params = import_params(filename)

    # Set calibration parameters
    calib_file = params["Calibration file"]
    mtx, dist, calib_w, calib_h = update_calibration(calib_file)

    # Set camera parameters
    cam_input = params["Camera input"] 
    cam_type = params["Camera type"]
    cam, color_correction_param, contrast_lut, gamma_lut = update_cam_input(cam_input, cam_type, calib_w, calib_h)

    # Get the list of image names in the folder
    images = os.listdir("experiment/slides")
    
    # Initialize JSON file
    experiment_data = {
        "Lens": calib_file,
        "Distance": 50,
        "Horizontal angle": 0,
        "Vertical angle": 0,
        "Lighting conditions": "Normal"
    }
    experiment_data_json = json.dumps(experiment_data, indent=4)
    with open("6_50_0_0_0.json", "w") as file:
        file.write(experiment_data_json)

    # # Shuffle the image names in a random order
    # random.shuffle(images)

    # Initialize OCR engines
    easyocr_reader = easyocr.Reader(['en'], gpu=False)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
    os.environ['TESSDATA_PREFIX'] = r".\ocr_script\Tesseract_sevenSegmentsLetsGoDigital\tessdata"

    # Show one slide to setup camera
    image_path = "experiment/slides/" + images[0]
    image = cv2.imread(image_path)
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow("slideshow_window", image)
    cv2.setWindowTitle("slideshow_window", "Slideshow")
    cv2.waitKey(1)

    # Show frame with crosshair to center camera on text
    while True:
        ret, frame = get_frame(cam, cam_type, color_correction_param, contrast_lut, gamma_lut)
        fh, fw = frame.shape[:2]
        new_fw, new_fh = resize_with_ratio(750, 750, fw, fh)
        frame = cv2.resize(frame, (new_fw, new_fh))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = draw_crosshair(frame)
        cv2.imshow("Center camera on text", frame)
        if cv2.waitKey(1) == 32:
            cv2.destroyWindow("Center camera on text")
            break

    # Print the shuffled image names
    for image in images:

        image_path = "experiment/slides/" + image
        img_code = image[:-4]
        if img_code[0] != "4":
            continue
        image = cv2.imread(image_path)
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        cv2.imshow("slideshow_window", image)
        cv2.setWindowTitle("slideshow_window", img_code)
        cv2.waitKey(1)

        print(f"Image {img_code}")

        natural_texts = []
        processed_texts = []
        previous_frame = np.zeros((2,2))
        frame = np.zeros((2,2))

        # Call OCR function 30 times with and without image processing per slide
        for i in tqdm(range(30)):  
            
            # Making sure the new frame is different from previous one
            while np.array_equal(previous_frame, frame):

                # Keep looking for frame with rectified perspective until found
                rectified_frame = None
                while rectified_frame is None:                
                    # Get new frame
                    ret, frame = get_frame(cam, cam_type, color_correction_param, contrast_lut, gamma_lut)
                    if not ret:
                        print("No frame could be found")
                        continue
                    # Rectify Frame
                    rectified_frame = rectify_image(frame, params, calib_w, calib_h, mtx, dist)
                
                if np.array_equal(previous_frame, frame):
                    print("Frame is same as previous")
            
            previous_frame = frame 
            
            if show_frame:
                fh, fw = frame.shape[:2]
                new_fw, new_fh = resize_with_ratio(750, 750, fw, fh)
                frame = cv2.resize(frame, (new_fw, new_fh))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("Cam frame", frame)


            # Get ROI for corresponding size
            roi_img = get_roi(rectified_frame, img_code, params["ROI list"])
            # Process image with corresponding pipeline
            processed_img = process_img(roi_img, img_code)

            # Display obsreved ROIs
            if display_rois:
                stacked_imgs = cv2.vconcat([roi_img, processed_img])
                h, w = stacked_imgs.shape[:2]
                new_w, new_h = resize_with_ratio(300, 500, w, h)
                stacked_imgs = cv2.resize(stacked_imgs, (new_w,new_h))
                stacked_imgs = cv2.cvtColor(stacked_imgs, cv2.COLOR_RGB2BGR)
                cv2.imshow("Observed", stacked_imgs)
                key = cv2.waitKey(1)
                if key == 32:
                    print("Going to next slide")
                    break
                elif key == 27:
                    print("Measurement exited before end.")
                    close_cam(cam, cam_type)
                    cv2.destroyAllWindows()
                    exit()

            # Call OCR
            natural_text = call_ocr(roi_img, img_code, easyocr_reader)
            natural_texts.append(natural_text)
            processed_text = call_ocr(processed_img, img_code, easyocr_reader)
            processed_texts.append(processed_text)
            # print(f"{img_code}_{i+1} => {natural_text}, {processed_text}")  
        
        # Store results in result array
        result = {
            "Image code": img_code,
            "Unproc img results": natural_texts,
            "Proc img results": processed_texts
            }

        # Write the JSON data to a file
        result_json = json.dumps(result, indent=4)
        with open("6_50_0_0_0.json", "a") as file:
            file.write(result_json)
      
    close_cam(cam, cam_type)
    cv2.destroyAllWindows()

    cycle_time = datetime.datetime.now() - start_time
    print(f"Total cycle duration = {cycle_time}")




if __name__ == '__main__':
    main()