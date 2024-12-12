import cv2
import numpy as np
import json
import os

def draw_bubbles_from_json(image_path, json_path):
    # Check if the JSON file exists
    if not os.path.exists(json_path):
        print("JSON file not found:", json_path)
        return None

    # Load the original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("Error loading image. Please check the file path or file integrity.")
        return None

    # Create a blank image with the same dimensions as the original image
    bubble_image = np.zeros_like(original_image)

    # Load the bubble information from the JSON file
    with open(json_path, 'r') as json_file:
        bubble_data = json.load(json_file)

    # Draw the bubble borders based on JSON data
    for bubble in bubble_data['bubbles']:
        contour_points = np.array(bubble['border'], dtype=np.int32)
        cv2.drawContours(bubble_image, [contour_points], -1, (0, 255, 0), 2)  # Draw contours in green

    # Save the bubble image
    bubble_image_path = os.path.splitext(image_path)[0] + '_bubble_image.png'
    cv2.imwrite(bubble_image_path, bubble_image)
    print(f"Bubble borders image saved as {bubble_image_path}")

    # Display the original image and the bubble image for comparison
    cv2.imshow('Original Image', original_image)
    cv2.imshow('Bubble Borders from JSON', bubble_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = r'fl_0_0470_0_xz.png'  # Path to the original image
json_path = os.path.splitext(image_path)[0] + '_bubble_info.json'  # Corresponding JSON file path
draw_bubbles_from_json(image_path, json_path)
