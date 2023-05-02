import numpy as np
import cv2 as cv
import glob
from tqdm import tqdm


camera = "maikes_phone"
imgs_path = "cam_calibration/cameras/" + camera
square_len = 0.0235
chess_width = 9
chess_height = 6

chessboardSize = (chess_width,chess_height)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2) * square_len 

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(imgs_path  + '/*.png') + glob.glob(imgs_path  + '/*.jpg')
print(f"{len(images)} found")

for image in tqdm(images):
    
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
 
        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        # Resize image
        img_height, img_width = img.shape[:2]
        max_width = 1200
        max_height = 1080
        scale_factor = min(max_width / img_width, max_height / img_height)
        img = cv.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_AREA)
        cv.imshow('Calibration image', img)
        cv.waitKey(50)

cv.destroyAllWindows()
        
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Reprojection Error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )