import cv2
import os
import numpy as np
from sklearn.cluster import KMeans

# Paths to input folder and output folder for unique color cars
input_folder = 'distinct_cars'
output_folder = 'unique_color_cars'
os.makedirs(output_folder, exist_ok=True)

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

# Function to calculate the Euclidean distance between two RGB colors
def color_distance(color1, color2):
    return np.linalg.norm(color1 - color2)

# List to store unique colors and their corresponding images
unique_colors = []
car_count = 0

# Process each image in the distinct_cars folder
for image_file in sorted(os.listdir(input_folder)):
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)

    # Calculate the dominant color of the current image
    dominant_color = get_dominant_color(image)

    # Check if this dominant color is similar to any previously saved unique color
    is_unique_color = True
    for color in unique_colors:
        if color_distance(dominant_color, color) < color_similarity_threshold:
            is_unique_color = False
            break

    # If the color is unique, save the image and add the color to the list
    if is_unique_color:
        unique_colors.append(dominant_color)
        car_count += 1
        output_path = os.path.join(output_folder, f'unique_color_car_{car_count:04d}.jpg')
        cv2.imwrite(output_path, image)

print(f"Color-based filtering complete. Total unique color cars saved: {car_count}")
print("Unique color car images are saved in:", output_folder)
