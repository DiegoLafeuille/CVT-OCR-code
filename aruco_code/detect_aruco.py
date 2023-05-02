import numpy as np
import cv2
from cv2 import aruco
import pickle

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

camera = "diegos_iriun"
aruco_dict = ARUCO_DICT["DICT_4X4_50"]
cam_input = 1
aruco_size = 0.04 # in meters


# Get calibration parameters

# Path to pickle file
calib_file_path = "cam_calibration/cameras/" + camera + "/calibration_params.pickle"

# Load the calibration parameters from the pickle file
with open(calib_file_path, 'rb') as f:
    calibration_params = pickle.load(f)

# Extract the parameters from the dictionary
mtx = calibration_params["mtx"]
dist = calibration_params["dist"]
calib_w = calibration_params["calib_w"]
calib_h = calibration_params["calib_h"]


if any(x is None for x in (mtx, dist)):
    print("Failed to retrieve calibration parameters.")


def track(matrix, distort, cam_input, aruco_dict, aruco_size):
    
    # Set up video capture from default camera
    cap = cv2.VideoCapture(cam_input)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, calib_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, calib_h)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Calib resolution = {calib_w}x{calib_h}")
    print(f"Actual resolution = {width}x{height}")
    

    while True:
        
        ret, frame = cap.read()

        if not ret:
            print('No image found')
            break

        # operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Change grayscale
        
        aruco_dict_def = cv2.aruco.getPredefinedDictionary(aruco_dict)
        
        # lists of ids and the corners beloning to each id
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict_def)        
        if np.all(ids is not None):  # If there are markers found by detector
            
            tvecs = []
            for i in range(0, len(ids)):  # Iterate in markers
                
                # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[i], aruco_size, matrix, distort)
                tvecs.append(tvec)
                (rvec - tvec).any()  # get rid of that nasty numpy value array error
                aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers
                cv2.drawFrameAxes(frame, matrix, distort, rvec, tvec, 0.1, 2)  # Draw Axis
                cv2.putText(frame, 
                           f"Dist: {round(np.linalg.norm(tvec[0][0]), 2)} m", 
                           corners[i][0][1].astype(int), 
                           cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2, 
                           cv2.LINE_AA,
                )

        # Resize image
        max_width = 1280
        max_height = 960
        scale = min(max_width / width, max_height / height)
        resized_frame = cv2.resize(frame, (int(width*scale), int(height*scale)), interpolation=cv2.INTER_AREA)

        # Display the resulting frame
        cv2.imshow('frame', resized_frame)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

track(mtx, dist, cam_input, aruco_dict, aruco_size)