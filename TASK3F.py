import cv2
import os
from skimage.metrics import structural_similarity as ssim

# Define paths for reference image, video, and output folder
reference_image_path = r"C:\Users\ASUS\Downloads\ronaldo3.jpg" 
video_path = r"C:\Users\ASUS\Downloads\Ronaldo real Madrid lineup 4k free clip _Clip for edits_.mp4"
output_folder = r"D:\FALL SEMESTER 2024-2025\CSE 4037 IMG & VIDEO ANALYTICS\LAB6\Task3_Output"

# Load Haar cascades for face, eyes, and mouth detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Load the reference image and convert it to grayscale
reference_image = cv2.imread(reference_image_path)
reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

# Detect face in the reference image
ref_faces = face_cascade.detectMultiScale(reference_gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
if len(ref_faces) == 0:
    print("No face detected in the reference image.")
else:
    x, y, w, h = ref_faces[0]
    reference_face = reference_gray[y:y+h, x:x+w]

    # Detect eyes and mouth within the reference face region
    ref_eyes = eye_cascade.detectMultiScale(reference_face, scaleFactor=1.1, minNeighbors=5)
    ref_mouth = mouth_cascade.detectMultiScale(reference_face, scaleFactor=1.7, minNeighbors=20)

# Open the video file
video = cv2.VideoCapture(video_path)

# Get the frame dimensions and frames per second (fps) of the input video
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Define output video path
output_video_path = os.path.join(output_folder, "output_with_opencv_similarity.mp4")

# Define the codec and create VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Process each frame in the video
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the current frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (fx, fy, fw, fh) in faces:
        detected_face = gray_frame[fy:fy+fh, fx:fx+fw]

        # Detect eyes and mouth within the detected face region
        eyes = eye_cascade.detectMultiScale(detected_face, scaleFactor=1.1, minNeighbors=5)
        mouth = mouth_cascade.detectMultiScale(detected_face, scaleFactor=1.7, minNeighbors=20)

        # Edge detection for feature comparison
        reference_face_edges = cv2.Canny(reference_face, 100, 200)
        detected_face_edges = cv2.Canny(detected_face, 100, 200)

        # Resize detected edges to match reference for SSIM comparison
        resized_edges = cv2.resize(detected_face_edges, (reference_face_edges.shape[1], reference_face_edges.shape[0]))
        face_score = ssim(reference_face_edges, resized_edges)

        # Feature matching based on eyes and mouth
        eye_match_score, mouth_match_score = 0, 0

        # Compare eye regions if detected
        if len(eyes) >= 2 and len(ref_eyes) >= 2:
            for (ex, ey, ew, eh), (rex, rey, rew, reh) in zip(eyes[:2], ref_eyes[:2]):
                eye_region = detected_face[ey:ey+eh, ex:ex+ew]
                ref_eye_region = reference_face[rey:rey+reh, rex:rex+rew]

                eye_edges = cv2.Canny(eye_region, 100, 200)
                ref_eye_edges = cv2.Canny(ref_eye_region, 100, 200)

                eye_resized = cv2.resize(eye_edges, (ref_eye_edges.shape[1], ref_eye_edges.shape[0]))
                eye_match_score += ssim(ref_eye_edges, eye_resized)

            eye_match_score /= 2  # Average score for eyes

        # Compare mouth region if detected
        if len(mouth) > 0 and len(ref_mouth) > 0:
            mx, my, mw, mh = mouth[0]
            rmx, rmy, rmw, rmh = ref_mouth[0]

            mouth_region = detected_face[my:my+mh, mx:mx+mw]
            ref_mouth_region = reference_face[rmy:rmy+rmh, rmx:rmx+rmw]

            mouth_edges = cv2.Canny(mouth_region, 100, 200)
            ref_mouth_edges = cv2.Canny(ref_mouth_region, 100, 200)

            mouth_resized = cv2.resize(mouth_edges, (ref_mouth_edges.shape[1], ref_mouth_edges.shape[0]))
            mouth_match_score = ssim(ref_mouth_edges, mouth_resized)

        # Calculate an overall score (weighted by importance)
        overall_score = (0.5 * face_score) + (0.25 * eye_match_score) + (0.25 * mouth_match_score)

        # Draw bounding box and add label only if the score exceeds 0.29
        if overall_score > 0.29:
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
            cv2.putText(frame, f"Match Found (Score: {overall_score:.2f})", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Write the processed frame to the output video
    output_video.write(frame)

# Release resources
video.release()
output_video.release()

print(f"Output video saved at: {output_video_path}")


