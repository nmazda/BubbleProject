import cv2
import numpy as np
import os
import json
import argparse
from collections import defaultdict

def detect_bubble_borders(input_dir, output_dir):
    """
    Detect bubble borders in images from the input directory,
    save the bubble information as JSON and the bubble border images.

    Args:
        input_dir (str): Directory containing input images.
        output_dir (str): Directory to save output images and JSON files.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to hold bubble information for each data/view combination
    bubble_data = defaultdict(lambda: defaultdict(lambda: {"chunks": []}))

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

        # Resize the image to 256x256
        img_resized = cv2.resize(img, (256, 256))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # Thresholding to create a binary image
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a white background for contour visualization
        contour_image = np.ones_like(img_resized) * 255

        # Extract data name, view, and chunk from the filename
        print(filename)
        data_name, chunk, view = filename.rsplit('_', 2)
        print(data_name)
        print(view)
        print(chunk)

        # Prepare chunk data
        chunk_data = {"chunk_id": int(chunk), "bubbles": []}

        # Process each contour found
        for i, contour in enumerate(contours):
            bubble_size = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)

            # Prepare outline information
            outline = {
                "min_x": int(x),
                "max_x": int(x + w),
                "min_y": int(y),
                "max_y": int(y + h)
            }

            contour_points = contour[:, 0, :].tolist()
            cv2.drawContours(contour_image, contours, i, (0, 0, 0), 2)

            # Append bubble information to the current chunk
            chunk_data["bubbles"].append({
                "bubble_id": f"bubble_{i}",
                "bubble_size": bubble_size,
                "outline": outline,
                "border": contour_points
            })

        # Add chunk data to the respective data/view entry
        bubble_data[data_name][view]["chunks"].append(chunk_data)

        # Save the contour image
        contour_image_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_bubble_borders.png")
        cv2.imwrite(contour_image_path, contour_image)

        print(f"Processed {filename}: Image saved to {contour_image_path}")

    # Save bubble information to a single JSON file per data/view combination
    for data_name, views in bubble_data.items():
        for view, info in views.items():
            json_save_path = os.path.join(output_dir, f"{data_name}_{view}_bubble_info.json")
            with open(json_save_path, 'w') as json_file:
                json.dump(info, json_file, indent=4)
            print(f"Bubble info saved to {json_save_path}")

def main():
    # Set up argument parser for command line input
    parser = argparse.ArgumentParser(description='Detect bubble borders in images.')
    parser.add_argument('--input_dir', type=str, help='Directory containing input images.')
    parser.add_argument('--output_dir', type=str, help='Directory to save output images and JSON files.')

    # Parse the command line arguments
    args = parser.parse_args()

    # Run the bubble detection function
    detect_bubble_borders(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
