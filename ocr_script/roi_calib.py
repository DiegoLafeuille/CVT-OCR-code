import cv2
import numpy as np
from sympy import symbols, Eq
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

class Surface:
    
    def __init__(self, aruco_dict, board, world_coords):
        
        self.aruco_dict = aruco_dict
        self.board = board
        self.world_coords = world_coords
        # self.recognized_counter = 0
        self.update_surface_dimensions()

    def update_surface_dimensions(self):
        '''Calculates surface dimensions based on its world coordinates'''

        # Gets max width of upper and lower surface sides
        width_AD = np.linalg.norm(self.world_coords[0] - self.world_coords[3])
        width_BC = np.linalg.norm(self.world_coords[1] - self.world_coords[2])
        self.width = max(width_AD, width_BC)

        # Gets max height of left and right surface sides
        height_AB = np.linalg.norm(self.world_coords[0] - self.world_coords[1])
        height_CD = np.linalg.norm(self.world_coords[2] - self.world_coords[3])
        self.height = max(height_AB, height_CD)

        # # Resize for max image size within original size while keeping surface ratio
        # new_width, new_height = resize_with_ratio(calib_w, calib_h, width, height)



def correct_perspective(frame, surface, mtx, dist, last_rvec, last_tvec):
    '''Warps the image to correct the perspective and zoom in on the target surface'''
    
    # Get frame dimensions
    frame_h, frame_w = frame.shape[:2]

    # Detect markers
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, surface.aruco_dict)
    cv2.aruco.refineDetectedMarkers(gray, surface.board, corners, ids, rejectedImgPoints)

    # Get Charuco pose estimation if markers found
    retval = False
    if np.all(ids is not None):
        charucoretval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, surface.board)
        frame = cv2.aruco.drawDetectedCornersCharuco(frame, charucoCorners, charucoIds, (0,255,0))
        retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, surface.board, mtx, dist, np.zeros((3, 1)), np.zeros((3, 1)))
    
    # If Charuco board has never been detected, return orifinal frame
    if not retval and last_rvec is None and last_tvec is None:
        return frame, last_rvec, last_tvec

    # Calculate new pose for smoother transitions
    if retval:
        last_rvec, last_tvec = new_pose(last_rvec, last_tvec, rvec, tvec)
    
    # If woorld coordinates of target surface unknown, draw detected board on frame
    if not surface.world_coords and last_rvec is not None and last_tvec is not None:
        frame = cv2.drawFrameAxes(frame, mtx, dist, last_rvec, last_tvec, 0.1)
        return frame, last_rvec, last_tvec

    # Get image coordinates of target surface
    img_coords = [get_point_img_coordinates(point_coords, last_rvec, last_tvec, mtx, dist) for point_coords in surface.world_coords]
    
    # Prepare warp input points
    warp_input_pts = []
    for point in img_coords:
        point_canvas_coords = [int(point[0]), int(point[1])]
        warp_input_pts.append(point_canvas_coords)
    warp_input_pts = np.float32(warp_input_pts)

    # Resize surface dimensions to fit original frame dimensions while preserving surface ratio 
    new_width, new_height = resize_with_ratio(frame_w, frame_h, surface.width, surface.height)

    # If image coordinates are found, compute the perspective transform M and warp frame
    if len(warp_input_pts) > 0:
        warp_output_pts = np.float32([[0, 0],
                                [0, new_height - 1],
                                [new_width - 1, new_height - 1],
                                [new_width - 1, 0]])
        M = cv2.getPerspectiveTransform(warp_input_pts,warp_output_pts)
        corrected_frame = cv2.warpPerspective(frame,M,(new_width, new_height),flags=cv2.INTER_CUBIC)

    return corrected_frame, last_rvec, last_tvec
    
def new_pose(last_rvec, last_tvec, rvec, tvec):
    '''Updates Charuco board's pose estimation for smoother transitions'''
                    
    # First pose estimation
    if last_rvec is None or last_tvec is None:
        new_rvec = rvec
        new_tvec = tvec
    
    # Update pose estimation (0.8 * new + 0.2 * old)
    else:
        new_rvec = 0.8 * rvec + 0.2 * last_rvec
        new_tvec = 0.8 * tvec + 0.2 * last_tvec
    
    return new_rvec, new_tvec

def get_point_img_coordinates(point_world_coords, rvec, tvec, mtx, dist):
    '''Calculates the image coordinates of a single point'''
                
    point_3d = np.array(point_world_coords).reshape((1, 1, 3))

    # Use projectPoints to project the 3D point onto the 2D image plane
    point_2d, _ = cv2.projectPoints(point_3d, rvec, tvec, mtx, dist)

    # Extract the pixel coordinates of the projected point
    img_coords = tuple(map(int, point_2d[0, 0]))
    
    return img_coords

def get_line_equation(point, mtx, rvec, tvec):
    '''Creates the equation in world coordinates of the line passing through 
    the camera and a point in the image. Output is a vector (X(s), Y(s), Z(s)).
    https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html'''

    s = symbols('s')
    x = np.array([[point[0]], [point[1]], [1]])

    rot_mat = cv2.Rodrigues(rvec)[0]
    inv_rodr = np.linalg.inv(rot_mat)
    
    inv_mtx = np.linalg.inv(mtx)

    line_equation = inv_rodr @ ((inv_mtx @ (s * x)) - tvec.reshape((3,1)))
    
    return line_equation

def get_lines_intersection(p1, p2, q1, q2):
    """
    Given two lines, each represented by a pair of 3D pointsthey pass through, 
    computes the coordinates of the middle point of the shortest segment linking the two lines.
    """

    # Calculate direction vectors for each line
    p_dir = p2 - p1
    q_dir = q2 - q1

    # Calculate the translation between the two points of origin
    orig_translation = p1 - q1

    a = np.dot(p_dir, p_dir.T)
    b = np.dot(p_dir, q_dir.T)
    c = np.dot(q_dir, q_dir.T)
    d = np.dot(p_dir, orig_translation.T)
    e = np.dot(q_dir, orig_translation.T)
    denom = a*c - b*b

    if denom != 0:
        s = (b*e - c*d) / denom
        t = (a*e - b*d) / denom
        result = 0.5 * (p1 + s*p_dir + q1 + t*q_dir)
        result = [float(x) for x in result]
        return result
    
    # If lines are parallel (very improbable)
    else:
        return np.nan * np.ones(3)

def get_point_world_coords(line_eqs):
    """
    Given a list of lines, find the point that is closest to all the lines.
    The average of calculated intersections is used here for each point if 
    the surface has been indicated more than twice.
    """

    s = symbols('s')
    
    # Compute the intersection points of all pairs of lines
    intersections = []
    for i in range(len(line_eqs)):
        
        # # Print to see the line equations
        # print([f"{x}= {eq[0]}" for x, eq in zip(["X","Y","Z"], line_eqs[i])])   

        for j in range(i+1, len(line_eqs)):

            # Replace s in each line's system of equation X(s), Y(s), Z(s), to get two points per line
            p1 = np.array([eq[0].subs(s, 0) for eq in line_eqs[i]]).reshape((3))
            p2 = np.array([eq[0].subs(s, 1) for eq in line_eqs[i]]).reshape((3))

            q1 = np.array([eq[0].subs(s, 0) for eq in line_eqs[j]]).reshape((3))
            q2 = np.array([eq[0].subs(s, 1) for eq in line_eqs[j]]).reshape((3))
            
            # Get woorld coordinates of intersection point
            intersection = get_lines_intersection(p1, p2, q1, q2)
            intersections.append(intersection)

    # In case more than two lines have been indicated per surface corner, calculates the average of the resulting points
    x = np.mean(intersections, axis=0)

    # Return the solution as a point
    return x

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
