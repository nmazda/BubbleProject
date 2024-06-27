import os
import cv2
from PIL import Image

def convert_to_canny(image_path, canny_output_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert to black and white (gray scale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    # Convert to canny edges
    edges = cv2.Canny(gray, 100, 200)

    # Invert the edges (black to white and vice versa)
    edges = cv2.bitwise_not(edges)
    
    # Save the canny edge image
    cv2.imwrite(canny_output_path, edges)
    print(f"Processed {image_path} and saved raw and canny images.")

# Paths for the input and output
input_image_path = './test1.jpg'
output_canny_image_path = './canny_image.jpg'

# Ensure the output directories exist
os.makedirs(os.path.dirname(output_canny_image_path), exist_ok=True)

# Process the single image
convert_to_canny(input_image_path, output_canny_image_path)
