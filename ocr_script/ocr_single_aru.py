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
    # print(f"{calib_w}x{calib_h}")

    if any(x is None for x in (mtx, dist)):
        print("Failed to retrieve calibration parameters.")
        return None
    
    return mtx, dist, calib_w, calib_h

def main():
    # Load the image and camera parameters
    img = cv2.imread('c:/Users/DLafeuille/Desktop/BachelorArbeit/CVT-OCR-code/cam_calibration/cameras/maikes_phone/20230427_152149.jpg')
    camera_matrix, dist_coeffs, _, _ = update_cam("maikes_phone")

    # Define the ArUco dictionary and parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_size = 0.04

    # Detect the ArUco marker in the image
    corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict,)

    cv2.aruco.drawDetectedMarkers(img, corners) 

    if ids is not None:
        id1 = np.where(ids == 1)[0][0]
        
        # Estimate the pose of the marker
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[id1], marker_size, camera_matrix, dist_coeffs)

        cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, 0.1, 3)

        # Calculate the 3D points of the rectangle based on width, height, corner coordinates, and rotation angle
        width = 0.297  # width of rectangle in meters
        height = 0.210  # height of rectangle in meters
        x = marker_size/2  # x coordinate of upper left corner of rectangle in meters
        y = -marker_size/2  # y coordinate of upper left corner of rectangle in meters
        angle = 2  # rotation angle of rectangle in degrees
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



# def main():
#     # Load the image and camera parameters
#     img = cv2.imread('c:/Users/DLafeuille/Desktop/BachelorArbeit/CVT-OCR-code/cam_calibration/cameras/jans_webcam/WIN_20230414_09_38_46_Pro.jpg')
#     camera_matrix, dist_coeffs, _, _ = update_cam("jans_webcam")

#     # Define the ArUco dictionary and parameters
#     aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
#     marker_size = 0.08

#     # Detect the ArUco marker in the image
#     corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict,)

#     cv2.aruco.drawDetectedMarkers(img, corners) 

#     if ids is not None:
#         id1 = np.where(ids == 1)[0][0]
        
#         # Estimate the pose of the marker
#         rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[id1], marker_size, camera_matrix, dist_coeffs)

#         # Convert rotation vector to rotation matrix
#         rmat, _ = cv2.Rodrigues(rvec)

#         # Calculate the transformation matrix from marker to camera coordinates
#         marker_to_cam = np.hstack((rmat, tvec.T))
#         marker_to_cam = np.vstack((marker_to_cam, [0, 0, 0, 1]))

#         # Invert the transformation matrix to get the transformation matrix from camera to marker coordinates
#         cam_to_marker = np.linalg.inv(marker_to_cam)

#         # Define the rectangle points in world coordinates
#         width = 0.297  # width of rectangle in meters
#         height = 0.210  # height of rectangle in meters
#         x = marker_size/2  # x coordinate of center of rectangle in meters
#         y = -marker_size/2  # y coordinate of center of rectangle in meters
#         rect_points_3d_world = np.array([
#             [x - width/2, y + height/2, 0],
#             [x + width/2, y + height/2, 0],
#             [x + width/2, y - height/2, 0],
#             [x - width/2, y - height/2, 0]
#         ], dtype=np.float32)

#         # Transform the rectangle points from world coordinates to camera coordinates
#         rect_points_3d_cam = np.matmul(rect_points_3d_world, cam_to_marker[:3, :3].T) + cam_to_marker[:3, 3].T

#         # Project the 3D points onto the 2D image plane using the camera parameters
#         rect_points_2d, _ = cv2.projectPoints(rect_points_3d_cam, rvec, tvec, camera_matrix, dist_coeffs)

#         # Draw the detected rectangle on the image
#         img = cv2.polylines(img, [np.int32(rect_points_2d)], True, (0, 255, 0), 2)

#     img = cv2.resize(img, (int(1920/2),int(1080/2)))

#     cv2.imshow('Image', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
