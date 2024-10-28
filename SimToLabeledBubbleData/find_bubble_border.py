import cv2
import numpy as np
import os
import json

# Function to detect bubble borders and convert to 80x80
def detect_bubble_borders(image_path):
    # Check if the file exists
    if not os.path.exists(image_path):
        print("File not found:", image_path)
        return None, None

    # Load the image
    img = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if img is None:
        print("Error loading image. Please check the file path or file integrity.")
        return None, None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty image to draw contours
    contour_image = np.zeros_like(img)

    # Store contours for JSON
    bubble_borders = {}

    # Process each contour
    for i, contour in enumerate(contours):
        # Get the contour points and convert them to a list
        contour_points = contour[:, 0, :].tolist()  # Get x, y points

        # Draw contours on the contour_image
        cv2.drawContours(contour_image, contours, i, (0, 255, 0), 2)  # Draw in green with thickness of 2

        # Store the points in the dictionary
        bubble_borders[f'bubble_{i}'] = contour_points

    # Resize the contour image to 80x80
    contour_image_resized = cv2.resize(contour_image, (80, 80))

    # Save the resized contour image (optional)
    cv2.imwrite('bubble_borders_resized.png', contour_image_resized)

    # Save bubble borders data to JSON
    json_save_path = 'bubble_borders.json'
    with open(json_save_path, 'w') as json_file:
        json.dump(bubble_borders, json_file, indent=4)

    print(f"Bubble borders saved as {json_save_path}")

    return contours, contour_image_resized

# Example usage
image_path = r'fl_0_0470_0_xz.png'  # Update with your image path
contours, contour_image = detect_bubble_borders(image_path)

# Check if contours are detected and display the images
if contours is not None:
    cv2.imshow('Original Image', cv2.imread(image_path))
    cv2.imshow('Bubble Borders Resized', contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
