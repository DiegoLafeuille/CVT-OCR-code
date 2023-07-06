import cv2
import os
import re

# Set webcam name
webcam = "ELP"

# Open the camera
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3264)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2448)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"{width}x{height}")

# Specify the folder path where the images will be saved
folder_path = "cam_calibration/cameras/" + webcam + "/"

# Check if the folder exists, create it if necessary
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Get a list of existing image files in the folder
existing_images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Initialize the counter with the highest number from existing images
img_counter = 0
if existing_images:
    # Extract the image number from the filename and find the maximum
    img_numbers = [int(re.search(r"image_(\d+)\.png", f).group(1)) for f in existing_images if re.search(r"image_(\d+)\.png", f)]
    img_counter = max(img_numbers)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    # Show the frame in a window
    resized_frame = cv2.resize(frame,(int(width/4), int(height/4)))
    cv2.imshow("Camera", resized_frame)
    
    # Check if the user pressed the space bar
    if cv2.waitKey(1) == ord(' '):
        # Increment the image counter
        img_counter += 1
        
        # Save the image to disk
        filename = f"image_{img_counter}.png"
        cv2.imwrite(f"{folder_path}{filename}", frame)
        
        # Print a message to the console
        print(f"{filename} saved!")
        
    # Check if the user pressed the escape key to quit
    if cv2.waitKey(1) == 27:
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()