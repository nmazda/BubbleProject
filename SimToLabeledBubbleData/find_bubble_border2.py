import cv2
import numpy as np
import os
import json
import argparse

def detect_bubble_borders(input_dir, output_dir):
    """
    Detect bubble borders in images from the input directory,
    save the bubble information as JSON and the bubble border images.

    Args:
        input_dir (str): Directory containing input images.
        output_dir (str): Directory to save output JSON files.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each image in the input directory
    for filename in os.listdir(input_dir):
        image_path = os.path.join(input_dir, filename)

        # Check if the file is an image
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Skipping non-image file: {filename}")
            continue

        # Load the image
        img = cv2.imread(image_path)

        if img is None:
            print(f"Error loading image: {filename}.")
            continue

        # Get image dimensions
        height, width = img.shape[:2]

        # Process the entire image without resizing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Thresholding to create a binary image
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        shapes = []

        # Process each contour found
        for i, contour in enumerate(contours):
            # Convert contour to the required format
            points = contour.reshape(-1, 2).tolist()

            shape = {
                "label": "bubble",
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }

            shapes.append(shape)

        # Prepare the JSON data
        json_data = {
            "version": "4.5.6",
            "flags": {},
            "shapes": shapes,
            "imagePath": filename,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width
        }

        # Save bubble information to JSON
        json_save_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
        with open(json_save_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=2)

        print(f"Processed {filename}: Bubble info saved to {json_save_path}")

def main():
    # Set up argument parser for command line input
    parser = argparse.ArgumentParser(description='Detect bubble borders in images.')
    parser.add_argument('--input_dir', type=str, help='Directory containing input images.')
    parser.add_argument('--output_dir', type=str, help='Directory to save output JSON files.')

    # Parse the command line arguments
    args = parser.parse_args()

    # Run the bubble detection function
    detect_bubble_borders(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()