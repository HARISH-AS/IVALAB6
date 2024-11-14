import cv2
import os

# Path to the video
video_path = 'in.mp4'
output_folder = 'frames'

# Create a folder to save frames if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load the video
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
else:
    frame_count = 0
    
    # Loop through each frame in the video
    while cap.isOpened():
        ret, frame = cap.read()
        
        # If a frame is returned
        if ret:
            # Define filename for each frame
            frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
            
            # Save the frame as an image file
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
        else:
            break

    # Release the video capture object
    cap.release()
    print(f"Total frames saved: {frame_count}")
