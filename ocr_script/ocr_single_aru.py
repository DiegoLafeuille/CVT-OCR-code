import cv2
import numpy as np
import pickle



def update_cam(camera):
    # print(f"Updating camera: {camera}")

    # Path to pickle file
    calib_file_path = "cam_calibration/cameras/" + camera + "/calibration_params.pickle"

    # Load the calibration parameters from the pickle file
    with open(calib_file_path, 'rb') as f:
        calibration_params = pickle.load(f)

    # Extract the parameters from the dictionary
    mtx = calibration_params["mtx"]
    dist = calibration_params["dist"]
    calib_w = int(calibration_params["calib_w"])
    calib_h = int(calibration_params["calib_h"])
    print(f"{calib_w}x{calib_h}")

    if any(x is None for x in (mtx, dist)):
        print("Failed to retrieve calibration parameters.")
        return None
    
    return mtx, dist, calib_w, calib_h

def main():
    # Load the image and camera parameters
    img = cv2.imread('c:/Users/DLafeuille/Desktop/BachelorArbeit/CVT-OCR-code/ocr_script/client_display_maike.jpeg')
    print(f"{img.shape[1]}x{img.shape[0]}")
    camera_matrix, dist_coeffs, _, _ = update_cam("maikes_phone")

    # Define the ArUco dictionary and parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50)
    marker_size = 0.08

    # Detect the ArUco marker in the image
    corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict,)

    cv2.aruco.drawDetectedMarkers(img, corners) 

    if ids is not None:
        id1 = np.where(ids == 1)[0][0]
        
        # Estimate the pose of the marker
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[id1], marker_size, camera_matrix, dist_coeffs)

        cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, 0.1, 3)

        # Calculate the 3D points of the rectangle based on width, height, corner coordinates, and rotation angle
        width = 0.215   # width of rectangle in meters
        height = 0.15  # height of rectangle in meters
        x = -marker_size/2-0.01  # x coordinate of upper left corner of rectangle in meters
        y = -marker_size/2-0.02  # y coordinate of upper left corner of rectangle in meters
        angle = -91.5  # rotation angle of rectangle in degrees
        rect_points_3d = np.array([[x, y, 0], [x + width*np.cos(np.deg2rad(angle)), y + width*np.sin(np.deg2rad(angle)), 0], [x + height*np.sin(np.deg2rad(angle)) + width*np.cos(np.deg2rad(angle)), y - height*np.cos(np.deg2rad(angle)) + width*np.sin(np.deg2rad(angle)), 0], [x + height*np.sin(np.deg2rad(angle)), y - height*np.cos(np.deg2rad(angle)), 0]], dtype=np.float32)


        # Project the 3D points onto the 2D image plane using the camera parameters
        rect_points_2d, _ = cv2.projectPoints(rect_points_3d, rvec, tvec, camera_matrix, dist_coeffs)

        # Draw the detected rectangle on the image
        img = cv2.polylines(img, [np.int32(rect_points_2d)], True, (0, 255, 0), 2)

    img = cv2.resize(img, (int(1920/2),int(1080/2)))

    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

