import cv2
   

# Specify the path to your image
# image_path = "aruco_patterns\charuco_boards\charuco_20x15_DICT_4X4_1000_sl20_ml14.png"
image_path = "aruco_patterns\charuco_boards\charuco_40x30_DICT_4X4_1000_sl10_ml7.png"

# Load the image
image = cv2.imread(image_path)
print(image.shape[:2])

# Check if the image was loaded successfully
if image is None:
    print(f"Error: Unable to load image '{image_path}'")
    exit()


# Create a window to display the image
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 1280, 960)

# Display the image in the window
cv2.imshow("Image", image)
cv2.waitKey(0)
   

cv2.destroyAllWindows()

