import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import KMeans

# Paths to the input and output folders
input_folder = 'unique_cars'
distinct_folder = 'distinct_cars'
os.makedirs(distinct_folder, exist_ok=True)

# Threshold for SSIM similarity to consider two images as duplicates
similarity_threshold = 0.4

# Minimum dimensions for an image to be considered distinct
min_width, min_height = 200, 150

# Set a common size for all images for SSIM comparison
fixed_size = (200, 200)

# Define a threshold for color similarity (Euclidean distance in RGB space)
color_similarity_threshold = 50

# Function to get the dominant color of an image
def get_dominant_color(image, k=1):
    # Resize image to speed up processing
    img = cv2.resize(image, (100, 100))
    img = img.reshape((-1, 3))

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(img)
    dominant_color = kmeans.cluster_centers_[0]
    return dominant_color

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

# Function to calculate the Euclidean distance between two RGB colors
def color_distance(color1, color2):
    return np.linalg.norm(color1 - color2)

# List to store unique images and their dominant colors
distinct_images = []
distinct_colors = []
distinct_count = 0

# Process each image in the unique_cars folder
for image_file in sorted(os.listdir(input_folder)):
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)

    # Skip images that are smaller than the specified dimensions
    if image.shape[1] < min_width or image.shape[0] < min_height:
        continue

    # Calculate the dominant color of the current image
    dominant_color = get_dominant_color(image)

    # Compare with existing distinct images
    is_distinct = True
    for i, distinct_image in enumerate(distinct_images):
        # Check if images are similar based on SSIM
        if are_images_similar(image, distinct_image):
            # Check if dominant colors are also similar
            if color_distance(dominant_color, distinct_colors[i]) < color_similarity_threshold:
                is_distinct = False
                break

    # If image is distinct, add it to the distinct images list and save it
    if is_distinct:
        distinct_images.append(image)
        distinct_colors.append(dominant_color)
        distinct_count += 1
        output_path = os.path.join(distinct_folder, f'distinct_car_{distinct_count:04d}.jpg')
        cv2.imwrite(output_path, image)

print(f"Filtering complete. Total distinct cars saved: {distinct_count}")
print("Distinct car images are saved in:", distinct_folder)
