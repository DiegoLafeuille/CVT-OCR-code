# Imports
import numpy as np
import cv2
from cv2 import aruco
import pickle

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


camera = "jans_webcam"
cam_input = 0 # 0 for webcam, 1 for phone
aruco_dict = ARUCO_DICT["DICT_7X7_50"]
aruco_size = 0.079375 # in meters


def unwarp(camera, matrix, distort, cam_input, aruco_dict, aruco_size):

    # Get calibration parameters

    # Path to pickle file
    calib_file_path = "../cam_calibration/cameras/" + camera + "/calibration_params.pickle"

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


    # Set up video capture from default camera
    cap = cv2.VideoCapture(cam_input)

    # ret, frame = cap.read()
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolution = {width}x{height}")

    roi_corners = {}

    while True:
        
        ret, frame = cap.read()

        if not ret:
            print('No image found')
            break

        # operations on the frame come here
        frame = cv2.resize(frame, (calib_w,calib_h))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Change grayscale
        
        aruco_dict_def = cv2.aruco.getPredefinedDictionary(aruco_dict)
        
        # lists of ids and the corners beloning to each id
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict_def)
        
        if np.all(ids is not None):  # If there are markers found by detector

            tvecs = []
            rvecs = []
            for id in ids:  # Iterate in markers
                index = np.where(ids == id)[0][0]
                if id == 1:
                    roi_corners["A"] = [int(corners[index][0][2][0]), int(corners[index][0][2][1])] # Bottom right corner of ID 1
                if id == 2:
                    roi_corners["D"] = [int(corners[index][0][3][0]), int(corners[index][0][3][1])] # Bottom left corner of ID 2
                if id == 3:
                    roi_corners["C"] = [int(corners[index][0][0][0]), int(corners[index][0][0][1])] # Top left corner of ID 3
                if id == 4:
                    roi_corners["B"] = [int(corners[index][0][1][0]), int(corners[index][0][1][1])] # Top right corner of ID 4

                # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[index], aruco_size, matrix, distort)
                tvecs.append(tvec)
                rvecs.append(rvec)
                (rvec - tvec).any()  # get rid of that nasty numpy value array error

                if not all(id in ids for id in [1,2,3,4]):
                    cv2.drawFrameAxes(frame, matrix, distort, rvec, tvec, 0.1, 2)  # Draw Axis
                    for corner in corners:
                        cv2.putText(frame, 
                                f"Dist: {round(np.linalg.norm(tvec[0][0]), 2)} m", 
                                corner[0][1].astype(int), 
                                cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2, 
                                cv2.LINE_AA,
                        )

                                    
            if all(id in ids for id in [1,2,3,4]):

                indexA = np.where(ids == 1)[0][0]
                indexB = np.where(ids == 4)[0][0]
                indexC = np.where(ids == 3)[0][0]
                indexD = np.where(ids == 2)[0][0]

                width_AB = np.linalg.norm(tvecs[indexA]-tvecs[indexB]) - aruco_size
                width_CD = np.linalg.norm(tvecs[indexC]-tvecs[indexD]) - aruco_size
                corners_width = max(width_AB, width_CD)

                height_AD = np.linalg.norm(tvecs[indexA]-tvecs[indexD]) - aruco_size
                height_BC = np.linalg.norm(tvecs[indexB]-tvecs[indexC]) - aruco_size
                corners_height = max(height_AD, height_BC)


                # Resize image
                max_width = 1280
                max_height = 960
                scale = corners_height / corners_width
                # scale = min(max_width / corners_width, max_height / corners_height)

                if max_height * scale > max_width:
                    new_width = max_width
                    new_height = int(max_width / scale)
                else:
                    new_height = max_height
                    new_width = int(height * scale)

                input_pts = np.float32([roi_corners["A"], roi_corners["B"], roi_corners["C"], roi_corners["D"]])
                output_pts = np.float32([[0, 0],
                                        [0, new_height - 1],
                                        [new_width - 1, new_height - 1],
                                        [new_width - 1, 0]])
                
                # Compute the perspective transform M
                M = cv2.getPerspectiveTransform(input_pts,output_pts)
                frame = cv2.warpPerspective(frame,M,(new_width, new_height),flags=cv2.INTER_LINEAR)
            
            else:
                aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers
                # Resize image
                max_width = 1280
                max_height = 960
                scale = min(max_width / width, max_height / height)
                
                frame = cv2.resize(frame, (int(width*scale), int(height*scale)), interpolation=cv2.INTER_AREA)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()