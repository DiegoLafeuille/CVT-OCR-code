import cv2

# Open the camera
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1800)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"{width}x{height}")

# Initialize a variable to store the image
img_counter = 0

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
        cv2.imwrite(filename, frame)
        
        # Print a message to the console
        print(f"{filename} saved!")
        
    # Check if the user pressed the "q" key to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()