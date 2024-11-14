import cv2
import pandas as pd
import os
import numpy as np

# Define paths
video_path = r"C:\Users\ASUS\Downloads\mall1.mp4"
output_folder = r"D:\FALL SEMESTER 2024-2025\CSE 4037 IMG & VIDEO ANALYTICS\LAB6\Task5_Output"
output_excel_path = os.path.join(output_folder, "dwelling_times.xlsx")

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load the video
video = cv2.VideoCapture(video_path)
fps = video.get(cv2.CAP_PROP_FPS)

# Display the first frame to select the ROI
ret, first_frame = video.read()
if not ret:
    print("Failed to read video")
    video.release()
    exit()

# Select ROI manually by drawing a bounding box on the first frame
print("Select ROI on the video frame and press Enter/Space key to confirm.")
roi = cv2.selectROI("Select ROI", first_frame, showCrosshair=True, fromCenter=False)
cv2.destroyWindow("Select ROI")

# Extract ROI coordinates
roi_x, roi_y, roi_w, roi_h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])

# Initialize background subtractor for motion detection
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# Parameters for tracking
object_id = 0
tracking_objects = {}
dwelling_times = []

# Conversion factor: 1/4 second = 1 minute in time-lapse scale
time_lapse_factor = 4

# Frame index to track frame count
frame_index = 0

# Process video frames
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Convert to grayscale for processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)
    
    # Threshold and contours to detect objects
    _, thresh = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the ROI on the frame for visualization
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)

    # Process each detected contour
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue  # Skip small areas (noise)

        x, y, w, h = cv2.boundingRect(contour)
        
        # Check if the detected object is within the ROI
        if roi_x <= x <= roi_x + roi_w and roi_y <= y <= roi_y + roi_h:
            # Track objects within the ROI
            match_found = False
            for obj_id, data in tracking_objects.items():
                ox, oy, enter_frame = data
                if abs(x - ox) < 50 and abs(y - oy) < 50:  # Simple distance threshold for same object
                    tracking_objects[obj_id] = (x, y, enter_frame)  # Update position
                    match_found = True
                    break

            if not match_found:
                # New object detected
                tracking_objects[object_id] = (x, y, frame_index)
                object_id += 1

    # Calculate dwelling time for objects that left the ROI
    for obj_id, data in list(tracking_objects.items()):
        ox, oy, enter_frame = data
        if ox < roi_x or ox > roi_x + roi_w or oy < roi_y or oy > roi_y + roi_h:
            # Object has left the ROI
            time_in_frames = frame_index - enter_frame
            time_in_minutes = (time_in_frames / fps) * time_lapse_factor

            dwelling_times.append({
                "Object ID": obj_id,
                "Entry Time (Frame)": enter_frame,
                "Exit Time (Frame)": frame_index,
                "Dwelling Time (Minutes)": time_in_minutes
            })
            del tracking_objects[obj_id]

    frame_index += 1

    # Display the video with ROI and detected contours
    cv2.imshow("Video with ROI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video resources
video.release()
cv2.destroyAllWindows()

# Remaining objects still in ROI at the end of the video
for obj_id, data in tracking_objects.items():
    ox, oy, enter_frame = data
    time_in_frames = frame_index - enter_frame
    time_in_minutes = (time_in_frames / fps) * time_lapse_factor

    dwelling_times.append({
        "Object ID": obj_id,
        "Entry Time (Frame)": enter_frame,
        "Exit Time (Frame)": frame_index,
        "Dwelling Time (Minutes)": time_in_minutes
    })

# Save results to Excel
df = pd.DataFrame(dwelling_times)
df.to_excel(output_excel_path, index=False)

print(f"Dwelling times saved to: {output_excel_path}")


