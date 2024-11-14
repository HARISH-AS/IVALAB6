import cv2
import numpy as np
import os

# Define paths
video_path = r"C:\Users\ASUS\Downloads\task1final.mp4"  # Replace with the actual path to your video file
output_folder = r"D:\FALL SEMESTER 2024-2025\CSE 4037 IMG & VIDEO ANALYTICS\LAB6\Task1_Output"
os.makedirs(output_folder, exist_ok=True)

# Initialize video capture
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open the video.")
else:
    # Define the output path and video writer
    output_path = os.path.join(output_folder, "output_person_tagged.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Background subtractor for initial detection
    backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    # Initialize parameters for optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Tracking variables
    tracked_points = None
    old_gray = None

    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Initialize tracking in the first frame
        if tracked_points is None:
            # Apply background subtraction for initial motion detection
            fg_mask = backSub.apply(frame)

            # Threshold and remove noise
            _, thresh = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
            thresh = cv2.erode(thresh, None, iterations=2)
            thresh = cv2.dilate(thresh, None, iterations=2)

            # Find contours of moving objects
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Find the largest contour, assumed to be the person
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                tracked_points = np.array([[x + w // 2, y + h // 2]], dtype=np.float32).reshape(-1, 1, 2)
                old_gray = gray_frame

        else:
            # Use optical flow to track the detected person
            new_points, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, tracked_points, None, **lk_params)
            if new_points is not None and len(new_points) > 0:
                # Update tracked points and draw bounding box
                new_center = new_points[0][0].astype(int)
                cv2.rectangle(frame, (new_center[0] - 20, new_center[1] - 20),
                              (new_center[0] + 20, new_center[1] + 20), (0, 255, 0), 2)
                cv2.putText(frame, "Tracked Person", (new_center[0] - 20, new_center[1] - 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Update the tracking points
                tracked_points = new_points
                old_gray = gray_frame

        # Write the processed frame to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Output video with tagged and tracked person saved at {output_path}")
