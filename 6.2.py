import cv2
import os

# Paths to input and output folders
input_folder = 'frames'
output_folder = 'detected_cars_frames'
os.makedirs(output_folder, exist_ok=True)

# Initialize background subtractor for motion detection
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Loop through the saved frames
for frame_file in sorted(os.listdir(input_folder)):
    frame_path = os.path.join(input_folder, frame_file)
    frame = cv2.imread(frame_path)

    # Check if frame is loaded successfully
    if frame is None:
        continue

    # Apply background subtraction
    fg_mask = background_subtractor.apply(frame)
    
    # Threshold the mask to binary to get clean detection areas
    _, fg_mask = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)

    # Find contours to detect moving cars
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = False
    for cnt in contours:
        # Filter out small areas to ignore noise
        if cv2.contourArea(cnt) > 500:  # Adjust area threshold based on car size
            x, y, w, h = cv2.boundingRect(cnt)
            # Draw bounding box on the frame (optional for visualization)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            detected = True

    # If any cars were detected in the frame, save it
    if detected:
        output_path = os.path.join(output_folder, frame_file)
        cv2.imwrite(output_path, frame)

print("Car detection complete. Frames with detected cars are saved in:", output_folder)
