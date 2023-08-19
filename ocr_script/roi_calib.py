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
    
    def __init__(self, board):
        
        self.board = board
        self.img_coords = []
        self.line_equations = []
        self.world_coords = []
        self.recognized_counter = 0


    def get_line_equation(self, point, rvec, tvec):

        s = symbols('s')
        x = np.array([[point[0]], [point[1]], [1]])

        rot_mat = cv2.Rodrigues(rvec)[0]
        inv_rodr = np.linalg.inv(rot_mat)
        
        inv_mtx = np.linalg.inv(self.mtx)

        line_equation = inv_rodr @ ((inv_mtx @ (s * x)) - tvec.reshape((3,1)))
        self.line_equations.append(line_equation)

    def find_img_coords(self, rvec, tvec, mtx, dist):
            
        self.img_coords = []
        for point in self.world_coords:
            
            point_3d = np.array(point).reshape((1, 1, 3))

            # Use projectPoints to project the 3D point onto the 2D image plane
            point_2d, _ = cv2.projectPoints(point_3d, rvec, tvec, mtx, dist)

            # Extract the pixel coordinates of the projected point
            pixel_coords = tuple(map(int, point_2d[0, 0]))

            self.img_coords.append(pixel_coords)


def find_point_world_coords(self, line_eqs):
    """
    Given a list of lines, find the point that is closest to all the lines.
    The average of calculated intersections is used here for each point if 
    the surface has been indicated more than twice.
    """

    # Compute the intersection points of all pairs of lines
    intersections = []
    
    
    for i in range(len(line_eqs)):

        # print([f"{x}= {eq[0]}" for x, eq in zip(["X","Y","Z"], line_eqs[i])])   

        for j in range(i+1, len(line_eqs)):

            # Replace s in each line's system of equation X(s), Y(s), Z(s), to get two points
            p1 = np.array([eq[0].subs(self.s, 0) for eq in line_eqs[i]]).reshape((3))
            p2 = np.array([eq[0].subs(self.s, 1) for eq in line_eqs[i]]).reshape((3))
            
            q1 = np.array([eq[0].subs(self.s, 0) for eq in line_eqs[j]]).reshape((3))
            q2 = np.array([eq[0].subs(self.s, 1) for eq in line_eqs[j]]).reshape((3))
            
            intersection = self.lines_intersection(p1, p2, q1, q2)
            intersections.append(intersection)

    x = np.mean(intersections, axis=0)

    # Return the solution as a point
    return x







def correct_perspective(method, frame, aruco_dict, board, mtx, dist, frame_counter, last_rvec, last_tvec, surface_world_coords):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

    if method == "one_marker":
        
        frame_counter, last_rvec, last_tvec = new_pose_1m(gray, board, corners, ids, rejectedImgPoints, mtx, dist)
        
        if not surface_world_coords and last_rvec is not None and last_tvec is not None:
            frame = cv2.drawFrameAxes(frame, mtx, dist, last_rvec, last_tvec, 0.1)
            frame_counter = 0
            return frame, frame_counter

        warp_input_pts = get_surface_1m(last_rvec, last_tvec, surface_world_coords)
        
        #  Updating detected dimensions of object every 50 consecutive frames where surface is found
        if frame_counter % 50 == 0:
            new_width, new_height = get_surface_dims_1m()
        frame_counter += 1
    
    else:
        warp_input_pts = get_surface_4m()
        
        #  Updating detected dimensions of object every 50 consecutive frames where surface is found
        if frame_counter % 50 == 0:
            new_width, new_height = get_surface_dims_4m()
        frame_counter += 1
    
    # If surface coordinates are found
    if len(warp_input_pts) > 0:
        
        # Compute the perspective transform M and warp frame
        warp_output_pts = np.float32([[0, 0],
                                [0, new_height - 1],
                                [new_width - 1, new_height - 1],
                                [new_width - 1, 0]])

        M = cv2.getPerspectiveTransform(warp_input_pts,warp_output_pts)

        rectified_frame = cv2.warpPerspective(frame,M,(new_width, new_height),flags=cv2.INTER_CUBIC)
        # frame_height, frame_width = frame.shape[:2]

        new_canvas_width, new_canvas_height = resize_with_ratio(canvas_max_width, canvas_max_height, new_width, new_height)

        # Convert to RGB format
        frame = copy.copy(rectified_frame)
    
    return rectified_frame, frame_counter
    

def new_pose_1m(gray, board, corners, ids, rejectedImgPoints, mtx, dist, frame_counter):
    
    cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedImgPoints)
    retval = False

    # If there are markers found by detector
    if np.all(ids is not None):
        charucoretval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        frame = cv2.aruco.drawDetectedCornersCharuco(frame, charucoCorners, charucoIds, (0,255,0))
        retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, mtx, dist, np.zeros((3, 1)), np.zeros((3, 1)))
                
    # Initial pose estimation
    if retval and last_rvec is None and last_tvec is None:
        last_rvec = rvec
        last_tvec = tvec
    
    # Update pose estimation (0.8 * new + 0.2 * old) -> smoother transitions
    elif retval:
        last_rvec = 0.8 * rvec + 0.2 * last_rvec
        last_tvec = 0.8 * tvec + 0.2 * last_tvec
    
    else:
        frame_counter = 0
    
    return frame_counter, last_rvec, last_tvec


def get_surface_1m(last_rvec, last_tvec, surface_world_coords):
        
    warp_input_pts = []

    # Get surface image coordinates
    surface_img_coords = [find_img_coords(point_coords, last_rvec, last_tvec) for point_coords in surface_world_coords]
    
    for point in surface_img_coords:
        point_canvas_coords = [int(point[0]), int(point[1])]
        warp_input_pts.append(point_canvas_coords)
    warp_input_pts = np.float32(warp_input_pts)

    return warp_input_pts

def get_surface_dims_1m(surface_world_coords, calib_w, calib_h):

    width_AD = np.linalg.norm(surface_world_coords[0] - surface_world_coords[3])
    width_BC = np.linalg.norm(surface_world_coords[1] - surface_world_coords[2])
    surface_w = max(width_AD, width_BC)

    height_AB = np.linalg.norm(surface_world_coords[0] - surface_world_coords[1])
    height_CD = np.linalg.norm(surface_world_coords[2] - surface_world_coords[3])
    surface_h = max(height_AB, height_CD)


    # Resize for max image size within original size while keeping surface ratio
    new_width, new_height = resize_with_ratio(calib_w, calib_h, surface_w, surface_h)

    return new_width, new_height

def get_surface_dims_4m(tvecs, ids, calib_w, calib_h, aruco_size):
    
    indexA = np.where(ids == 1)[0][0]
    indexB = np.where(ids == 2)[0][0]
    # indexD = np.where(ids == 2)[0][0]
    indexC = np.where(ids == 3)[0][0]
    indexD = np.where(ids == 4)[0][0]
    # indexB = np.where(ids == 4)[0][0]

    width_AD = np.linalg.norm(tvecs[indexA]-tvecs[indexD]) - aruco_size
    width_BC = np.linalg.norm(tvecs[indexB]-tvecs[indexC]) - aruco_size
    surface_w = max(width_AD, width_BC)

    height_AB = np.linalg.norm(tvecs[indexA]-tvecs[indexB]) - aruco_size
    height_CD = np.linalg.norm(tvecs[indexC]-tvecs[indexD]) - aruco_size
    surface_h = max(height_AB, height_CD)

    new_width, new_height = resize_with_ratio(calib_w, calib_h, surface_w, surface_h)

    return new_width, new_height




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