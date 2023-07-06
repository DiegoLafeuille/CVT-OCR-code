import cv2
from tqdm import tqdm

def display_webcam():
    
    # Open the default webcam
    cap = cv2.VideoCapture(1)

    width = 3264
    height = 2448

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    
    ret, frame = cap.read()
    frame_h, frame_w = frame.shape[:2]
    if width != frame_w or height != frame_h:
        print("Resolution not matching")

    is_quit = False
    for i in range(0,-13, -1):
        
        if is_quit:
            break
        
        print(i)
        cap.set(cv2.CAP_PROP_EXPOSURE, i) 

        for _ in tqdm(range(0,100)):

            # Read a frame from the webcam
            ret, frame = cap.read()

            if not ret:
                print("No Frame found")
                continue

            # Display the frame
            frame = cv2.resize(frame, (width//4, height//4))
            cv2.imshow('Webcam Feed', frame)
            key = cv2.waitKey(1)

            if key == 32:
                print("Next exposure time")
                break

            if key == 27:
                is_quit = True
                print("Quitting script")
                break

    # Release the webcam and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


# Call the function to display the webcam feed
display_webcam()