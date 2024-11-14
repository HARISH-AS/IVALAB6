import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Paths to input images and output folder for unique cars
input_folder = 'categorized_cars'
unique_folder = 'unique_cars'
os.makedirs(unique_folder, exist_ok=True)

# Set minimum dimensions for an image to be considered
min_width, min_height = 100, 100  # Adjust based on your images

# Threshold for SSIM similarity to consider two images as duplicates
similarity_threshold = 0.8

# Set a common size for all images for SSIM comparison
fixed_size = (200, 200)

# Function to check similarity between two images using SSIM
def are_images_similar(img1, img2, threshold=similarity_threshold):
    # Resize both images to the fixed size
    img1_resized = cv2.resize(img1, fixed_size)
    img2_resized = cv2.resize(img2, fixed_size)
    
    # Convert images to grayscale
    img1_gray = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
    
    # Compute SSIM between the resized grayscale images
    s = ssim(img1_gray, img2_gray)
    return s > threshold

# List to store unique images
unique_images = []
car_count = 0

# Process each image in the input folder
for image_file in sorted(os.listdir(input_folder)):
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)

    # Skip images that are too small
    if image.shape[1] < min_width or image.shape[0] < min_height:
        continue

    # Compare with existing unique images
    is_unique = True
    for unique_image in unique_images:
        if are_images_similar(image, unique_image):
            is_unique = False
            break

    # If image is unique, add it to the unique images list and save it
    if is_unique:
        unique_images.append(image)
        car_count += 1
        output_path = os.path.join(unique_folder, f'unique_car_{car_count:04d}.jpg')
        cv2.imwrite(output_path, image)

print(f"Unique car detection complete. Total unique cars counted: {car_count}")
print("Unique car images are saved in:", unique_folder)
