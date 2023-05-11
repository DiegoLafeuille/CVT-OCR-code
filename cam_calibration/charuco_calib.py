import numpy as np
import cv2
import glob
from tqdm import tqdm
import argparse
import pickle



ap = argparse.ArgumentParser()
ap.add_argument("-c", "--camera", type=str, required=True,
	help="Name of camera geting calibrated")
ap.add_argument("-sl", "--square-length", type=float, default= 2,
	help="Length of one square in mm")
ap.add_argument("-ml", "--marker-length", type=float, default= 1.6,
	help="Length of one marker in mm")
ap.add_argument("-W", "--width", type=int, default= 12,
	help="Number of columns of board")
ap.add_argument("-H", "--height", type=int, default= 9,
	help="Number of rows of board")
ap.add_argument("-d", "--dictionary", type=str,
	default="DICT_4X4_1000",
	help="type of ArUCo tag to generate")
args = ap.parse_args()

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


camera = args.camera
imgs_path = "cam_calibration/cameras/" + camera
square_len = args.square_length
marker_len = args.marker_length
chess_width = args.width
chess_height = args.height
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args.dictionary])
board = cv2.aruco.CharucoBoard((chess_width, chess_height), square_len, marker_len, aruco_dict)



def read_chessboards(images):
    """
    Charuco base pose estimation.
    """
    print("POSE ESTIMATION STARTS:")
    allCorners = []
    allIds = []
    decimator = 0
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    for im in images:
        print("=> Processing image {0}".format(im))
        frame = cv2.imread(im)

        if frame.shape[:2] != (calib_h, calib_w):
            print("Calibration pictures have different resolutions!")
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

        if len(corners)>0:
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(gray, corner,
                                 winSize = (3,3),
                                 zeroZone = (-1,-1),
                                 criteria = criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(corners,ids,gray,board)
            if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%1==0:
                allCorners.append(res2[1])
                allIds.append(res2[2])

        decimator+=1

    imsize = gray.shape
    return allCorners,allIds,imsize

def calibrate_camera(allCorners,allIds,imsize):
    """
    Calibrates the camera using the dected corners.
    """
    print("CAMERA CALIBRATION")

    cameraMatrixInit = np.array([[ 1000.,    0., imsize[0]/2.],
                                 [    0., 1000., imsize[1]/2.],
                                 [    0.,    0.,           1.]])

    distCoeffsInit = np.zeros((5,1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    #flags = (cv2.CALIB_RATIONAL_MODEL)
    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=allCorners,
                      charucoIds=allIds,
                      board=board,
                      imageSize=imsize,
                      cameraMatrix=cameraMatrixInit,
                      distCoeffs=distCoeffsInit,
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors, perViewErrors



# Get calibration pictures
images = glob.glob(imgs_path + '/' + '*.' + 'jpg')
print(f"{len(images)} found")

# Get resolution of pictures
img = cv2.imread(images[0])
(calib_h, calib_w) = img.shape[:2]
print(f"Resolution: {calib_w}x{calib_h}")

allCorners,allIds,imsize=read_chessboards(images)
ret, mtx, dist, rvecs, tvecs, perViewErrors = calibrate_camera(allCorners,allIds,imsize)
print("ret = ", ret)

for i in range(len(images)):
    print(f"Reprojection error for image {images[i]}: {perViewErrors[i]}")

calibration_params = {"ret": ret, "mtx": mtx, "dist": dist, "rvecs": rvecs, "tvecs": tvecs, "calib_w": calib_w, "calib_h": calib_h}

# save the parameters in a pickle file
with open('cam_calibration/cameras/' + camera + '/calibration_params.pickle', 'wb') as f:
    pickle.dump(calibration_params, f)