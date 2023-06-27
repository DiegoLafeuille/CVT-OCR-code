import os
import random
import pandas as pd
import numpy as np
import pickle
import gxipy as gx
import cv2
import matplotlib.pyplot as plt
import copy

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
        
        try:
            device_manager = gx.DeviceManager()
            cam = device_manager.open_device_by_sn(cam_input)
            
            cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
            cam.ExposureTime.set(100000.0)
            cam.Gain.set(10.0)

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

def call_ocr():
    pass






def main():

    # Get GUI parameters for automated script
    filename = "controlled_experiment_params"
    params = import_params(filename)

    # Set calibration parameters
    calib_file = params["Calibration file"]
    mtx, dist, calib_w, calib_h = update_calibration(calib_file)

    # Set camera parameters
    cam_input = params["Camera input"] 
    cam_type = params["Camera type"]
    
    # cam, color_correction_param, contrast_lut, gamma_lut = update_cam_input(cam_input, cam_type, calib_w, calib_h)

    print(f"Changing camera to input {cam_input}")

    if cam_type == 'daheng':
        
        try:
            device_manager = gx.DeviceManager()
            cam = device_manager.open_device_by_sn(cam_input)
            
            cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
            cam.ExposureTime.set(100000.0)
            cam.Gain.set(10.0)

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


        except Exception as e:
            if str(e) == "FrameNotRead":
                print("Error: Unable to retrieve video feed from this camera.\n"
                                    + "Please check the connection and make sure the camera is not being used by another application.")
            else:
                print("Error: An error occurred while trying to retrieve video feed from this camera.")
                raise e

    # Get the list of image names in the folder
    images = os.listdir("experiment/slides")

    # # Shuffle the image names in a random order
    # random.shuffle(images)

    # Show one slide to setup camera
    image_path = "experiment/slides/" + images[0]
    image = cv2.imread(image_path)
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

    # Print the shuffled image names
    for image in images:

        image_path = "experiment/slides/" + image
        image = cv2.imread(image_path)
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cv2.imshow("Image", image)
        cv2.waitKey(1)

        for i in range(30):  

            print(f"OCR call number {i+1}")         

            rectified_frame = None
            while rectified_frame is None:                
                # Get new frame
                ret, frame = get_frame(cam, cam_type, color_correction_param, contrast_lut, gamma_lut)
                if not ret:
                    print("No frame could be found")
                    continue
                
                # Rectify Frame
                rectified_frame = rectify_image(frame, params, calib_w, calib_h, mtx, dist)
            
            call_ocr()
            


        # img_code = image[:-4]
        # gt_row = gt_df.loc[gt_df["Code"] == img_code]
        # ground_truth = gt_row['Ground truth'].values[0]
        # print(f"Code '{img_code}' -> {ground_truth}")

    cv2.destroyAllWindows()


    # # Import ground truths into dataframe
    # gt_df = pd.read_csv("experiment/ground_truths.csv", dtype=str)




if __name__ == '__main__':
    main()