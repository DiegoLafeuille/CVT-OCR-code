import cv2
import numpy as np

# Load the image and camera parameters
img = cv2.imread('image.jpg')
camera_matrix = np.loadtxt('camera_matrix.txt')
dist_coeffs = np.loadtxt('dist_coeffs.txt')

# Define the ArUco dictionary and parameters
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters_create()

# Detect the ArUco marker in the image
corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)

if ids is not None:
    # Estimate the pose of the marker
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)

    # Define the 3D points of the rectangle
    rect_points_3d = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    # Project the 3D points onto the 2D image plane using the camera parameters
    rect_points_2d, _ = cv2.projectPoints(rect_points_3d, rvecs, tvecs, camera_matrix, dist_coeffs)

    # Draw the detected rectangle on the image
    img = cv2.polylines(img, [np.int32(rect_points_2d)], True, (0, 255, 0), 2)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()