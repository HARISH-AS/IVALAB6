import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openpyxl import Workbook

# Define paths
video_path = r"C:\Users\ASUS\Downloads\task2final.mp4" # Path to your time-lapse video
output_folder = r"D:\FALL SEMESTER 2024-2025\CSE 4037 IMG & VIDEO ANALYTICS\LAB6\Output2"
os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist
output_video_path = os.path.join(output_folder, "task2_peak_shopping_timelapse.mp4")
output_excel_path = os.path.join(output_folder, "task2_peak_shopping_data.xlsx")

# Initialize video capture
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = None

# Background subtraction to detect people
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=30)


# Variables for tracking people count and intervals
frame_count = 0
people_count_over_time = []
interval_frame_samples = []  # To store frames of peak interval

# Process video in 2-second intervals (assuming 30 FPS for each frame in a 2-second interval)
fps = cap.get(cv2.CAP_PROP_FPS)
interval_frames = int(fps * 2)  # 2 seconds worth of frames
interval_index = 0  # Keep track of the interval

while cap.isOpened():
    people_count = 0  # Initialize people count for the interval

    # Process each 2-second interval
    for _ in range(interval_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Apply background subtraction to detect motion (i.e., people)
        fg_mask = backSub.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Count contours that meet size threshold (indicating a person)
        for contour in contours:
            if cv2.contourArea(contour) > 600:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                people_count += 1

        # Save one sample frame per interval for later review
        if interval_index == 0:
            interval_frame_samples.append(frame)

        # Write processed frame to output video
        if out is None:
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
        out.write(frame)

    # Append people count for this interval and update interval index
    people_count_over_time.append(people_count)
    interval_index += 1

    # Break loop if video ends
    if not ret:
        break

# Release resources
cap.release()
out.release()

# Identify peak shopping interval
peak_interval = np.argmax(people_count_over_time)
peak_count = people_count_over_time[peak_interval]
peak_time = peak_interval * 2  # Each interval represents 2 seconds

# Save data to Excel
wb = Workbook()
ws = wb.active
ws.title = "People Count Over Time"
ws.append(["Interval (2 seconds)", "People Count"])
for i, count in enumerate(people_count_over_time):
    ws.append([(i+1) * 2, count])
wb.save(output_excel_path)

# Plot and save the count data
plt.figure(figsize=(10, 6))
plt.plot([i*2 for i in range(len(people_count_over_time))], people_count_over_time, marker='o')
plt.xlabel('Time Interval (2-second segments)')
plt.ylabel('People Count')
plt.title('People Count Over Time - Peak Shopping Analysis')
plt.axvline(x=peak_interval*2, color='red', linestyle='--', label=f'Peak at {(peak_time)} sec')
plt.legend()
plt.savefig(os.path.join(output_folder, "task2_people_count_plot.png"))
plt.show()

# Print result summary
print(f"Output video saved at {output_video_path}")
print(f"Excel data saved at {output_excel_path}")
print(f"Peak shopping duration: Interval {peak_interval}, starting at {(peak_time)} seconds with {peak_count} people")


