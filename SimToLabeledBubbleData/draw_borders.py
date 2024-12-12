import cv2
import numpy as np
import os
import json
import argparse

def draw_bubble_borders(input_json_dir, input_real_dir, output_dir):
    """
    Draw bubble borders on images based on the information in JSON files.

    Args:
        input_json_dir (str): Directory containing JSON files with bubble information.
        input_real_dir (str): Directory containing input images.
        output_dir (str): Directory to save output images with bubble borders.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each JSON file in the input JSON directory
    for json_filename in os.listdir(input_json_dir):
        # Check if the file is a JSON file
        if not json_filename.lower().endswith('.json'):
            print(f"Skipping non-JSON file: {json_filename}")
            continue

        json_path = os.path.join(input_json_dir, json_filename)

        # Load bubble information from JSON
        with open(json_path, 'r') as json_file:
            bubble_data = json.load(json_file)

        # Extract the base name from the JSON file to find the corresponding image
        base_name = json_filename.replace('_bubble_info.json', '')
        image_name = f"{base_name}.png"  # Assuming images are in PNG format
        image_path = os.path.join(input_real_dir, image_name)

        # Load the corresponding image
        img = cv2.imread(image_path)

        if img is None:
            print(f"Error loading image: {image_name}.")
            continue

        # Draw bubble borders on the image
        for bubble in bubble_data['bubbles']:
            # Use the 'border' field for contour points
            contour_points = np.array(bubble['border'], dtype=np.int32)

            # Draw the contour in green color
            cv2.polylines(img, [contour_points], isClosed=True, color=(0, 255, 0), thickness=2)

        # Save the modified image with borders
        output_image_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_image_path, img)

        print(f"Processed {image_name}: Borders drawn and saved to {output_image_path}")


def main():
    # Set up argument parser for command line input
    parser = argparse.ArgumentParser(description='Draw bubble borders on images from JSON files.')
    parser.add_argument('--input_json_dir', type=str, help='Directory containing JSON files with bubble information.')
    parser.add_argument('--input_real_dir', type=str, help='Directory containing input images.')
    parser.add_argument('--output_dir', type=str, help='Directory to save output images with bubble borders.')

    # Parse the command line arguments
    args = parser.parse_args()

    # Run the bubble border drawing function
    draw_bubble_borders(args.input_json_dir, args.input_real_dir, args.output_dir)

if __name__ == "__main__":
    main()
