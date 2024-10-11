import argparse
from math import ceil
import cv2

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern-size',
                        '-p',
                        type=int,
                        nargs=2,
                        help='Size of the calibration pattern (width, height) in terms of squares',
                        required=True)
    parser.add_argument("-d", "--dictionary", type=str,
                    default="DICT_4X4_1000",
                    help="Dictionary of Aruco markers")
    parser.add_argument("-sl", "--square-length", type=int,
                    default=20,
                    help="Square length in mm")
    parser.add_argument("-ml", "--marker-length", type=int,
                    default=14,
                    help="Marker length in mm")
    parser.add_argument("-b", "--border", type=bool,
                    default="False",
                    help="Add a 15px white border around the Charuco board"
                    )
    args = parser.parse_args()

    # Create a Charuco board definition.
    PW = args.pattern_size[0]  # width of calibration pattern
    PH = args.pattern_size[1]  # height of calibration pattern 
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args.dictionary])
    square_length = args.square_length  # mm
    marker_length = args.marker_length  # mm
    board = cv2.aruco.CharucoBoard((PW, PH), square_length, marker_length, dictionary)

    pixels_per_cm = 100
    pixels_per_square = int(ceil(square_length / 10.0 * pixels_per_cm))
    img_board = board.generateImage((PW * pixels_per_square, PH * pixels_per_square), marginSize=0, borderBits=1)

    # Add a border of 15px if args.border is True
    if args.border:
        border_size = 15
        img_board = cv2.copyMakeBorder(
            img_board,
            top=border_size, bottom=border_size,
            left=border_size, right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )

    filedir = "./aruco_patterns/charuco_boards/"
    file = f"charuco_{PW}x{PH}_{args.dictionary}_sl{square_length}_ml{marker_length}.png"
    
    if not cv2.imwrite(filedir + file, img_board):
	    raise Exception("Could not write image")

if __name__ == '__main__':
    main()