import cv2
import numpy as np
import os

# Define paths
video_path = r"C:\Users\ASUS\Downloads\TASK4.mov"  # Update with the correct path if needed
output_folder = r"D:\FALL SEMESTER 2024-2025\CSE 4037 IMG & VIDEO ANALYTICS\LAB6\Task4_Output"
os.makedirs(output_folder, exist_ok=True)
output_video_path = os.path.join(output_folder, "output_task4.mp4")

# Load the video
cap = cv2.VideoCapture(video_path)

# Define the output video settings
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Define the Region of Interest (ROI) based on analysis for focused entrance detection
roi_start_x, roi_start_y = 120, 120  # Starting point of ROI
roi_width, roi_height = 240, 120  # Width and height of ROI

# Define motion detection thresholds and entry/exit zones
motion_threshold = 1500  # Minimum area of motion to consider significant
entry_threshold_y = roi_height * 0.6  # 60% of ROI height for entry detection
exit_threshold_y = roi_height * 0.3  # 30% of ROI height for exit detection

# Initialize counters and previous frame for motion detection
enter_count = 0
exit_count = 0
previous_frame = None

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale and isolate the ROI
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_frame = gray_frame[roi_start_y:roi_start_y + roi_height, roi_start_x:roi_start_x + roi_width]

    # Draw the ROI and detection lines on the frame
    cv2.rectangle(frame, (roi_start_x, roi_start_y), (roi_start_x + roi_width, roi_start_y + roi_height), (0, 255, 0), 2)
    entry_line_y = roi_start_y + int(entry_threshold_y)
    exit_line_y = roi_start_y + int(exit_threshold_y)
    cv2.line(frame, (roi_start_x, entry_line_y), (roi_start_x + roi_width, entry_line_y), (0, 255, 0), 2)  # Entry line
    cv2.line(frame, (roi_start_x, exit_line_y), (roi_start_x + roi_width, exit_line_y), (0, 0, 255), 2)  # Exit line

    # Initialize the previous frame if it hasn't been set yet
    if previous_frame is None:
        previous_frame = roi_frame
        continue

    # Calculate the difference between the current frame and the previous frame
    frame_diff = cv2.absdiff(previous_frame, roi_frame)
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
    previous_frame = roi_frame

    # Detect contours to find motion
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process each contour to track movement direction
    for contour in contours:
        if cv2.contourArea(contour) > motion_threshold:  # Only consider large motions
            x, y, w, h = cv2.boundingRect(contour)
            motion_center_y = y + h // 2  # Get the y-center of the motion

            # Draw bounding box for detected person
            cv2.rectangle(frame, (roi_start_x + x, roi_start_y + y), (roi_start_x + x + w, roi_start_y + y + h), (255, 0, 0), 2)

            # Check if the bounding box center crosses entry or exit lines
            if motion_center_y > entry_threshold_y:
                enter_count += 1
                cv2.putText(frame, "Entry", (roi_start_x + x, roi_start_y + y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            elif motion_center_y < exit_threshold_y:
                exit_count += 1
                cv2.putText(frame, "Exit", (roi_start_x + x, roi_start_y + y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Write the annotated frame to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()

# Display the final counts
print("Total Entries:", enter_count)
print("Total Exits:", exit_count)
print(f"Output video saved at: {output_video_path}")
