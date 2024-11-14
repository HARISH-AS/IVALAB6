import cv2
import os
import numpy as np

# Paths to input and output folders
input_folder = 'frames'
output_folder = 'categorized_cars'
os.makedirs(output_folder, exist_ok=True)

# Initialize background subtractor
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Define color ranges for various colors
color_ranges = {
    "red": ((0, 100, 100), (10, 255, 255)),
    "orange": ((10, 100, 100), (25, 255, 255)),
    "yellow": ((25, 100, 100), (35, 255, 255)),
    "green": ((35, 50, 50), (85, 255, 255)),
    "cyan": ((85, 100, 100), (95, 255, 255)),
    "blue": ((95, 150, 0), (125, 255, 255)),
    "purple": ((125, 100, 100), (150, 255, 255)),
    "pink": ((150, 100, 100), (170, 255, 255)),
    "white": ((0, 0, 200), (180, 30, 255)),  # Bright area with low saturation
    "gray": ((0, 0, 50), (180, 20, 200)),    # Low saturation, moderate brightness
    "black": ((0, 0, 0), (180, 255, 50)),    # Very low brightness
}

# Process each saved frame
car_count = 0
for frame_file in sorted(os.listdir(input_folder)):
    frame_path = os.path.join(input_folder, frame_file)
    frame = cv2.imread(frame_path)

    if frame is None:
        continue

    # Apply background subtraction to detect moving objects
    fg_mask = background_subtractor.apply(frame)
    _, fg_mask = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)

    # Find contours of moving objects
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # Filter out small contours to remove noise
        if cv2.contourArea(cnt) > 500:  # Adjust based on car size
            x, y, w, h = cv2.boundingRect(cnt)
            car_region = frame[y:y + h, x:x + w]

            # Convert to HSV for color-based classification
            hsv = cv2.cvtColor(car_region, cv2.COLOR_BGR2HSV)

            # Determine car color
            car_color = "unknown"
            for color_name, (lower, upper) in color_ranges.items():
                mask = cv2.inRange(hsv, lower, upper)
                if cv2.countNonZero(mask) > 500:  # Adjust threshold as needed
                    car_color = color_name
                    break

            # Save categorized frame with color label
            car_count += 1
            output_path = os.path.join(output_folder, f'{car_color}_car_{car_count:04d}.jpg')
            cv2.imwrite(output_path, car_region)

print("Car detection and categorization complete. Categorized frames are saved in:", output_folder)
