import os
import cv2
import argparse

def crop_and_resize_image(image, crop_coords, new_size):
    left, top, right, bottom = crop_coords
    cropped_image = image[top:bottom, left:right]
    resized_image = cv2.resize(cropped_image, new_size, interpolation=cv2.INTER_LINEAR)
    return resized_image

#changed below function to test on whole image instead of chunks of image
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crop and Resize Images')
    parser.add_argument('--input_dir', type=str, default='Uncropped_test', help='Input directory containing images')
    parser.add_argument('--output_dir', type=str, default='BW_test', help='Output directory to save processed images')
    parser.add_argument('--new_size', type=int, nargs=2, default=(256, 256), help='New size (width, height) for the resized images')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    new_size = tuple(args.new_size)

    crop_coords_xz = (114, 63, 282, 231)
    crop_coords_yz = (117, 63, 285, 231)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.png'):
            file_path = os.path.join(input_dir, file_name)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            if image is not None:
                if '_xz' in file_name:
                    crop_coords = crop_coords_xz
                elif '_yz' in file_name:
                    crop_coords = crop_coords_yz
                else:
                    print(f"Skipping {file_name}: Perspective not recognized")
                    continue

                cropped_resized_image = crop_and_resize_image(image, crop_coords, new_size)
                output_image_path = os.path.join(output_dir, file_name)
                cv2.imwrite(output_image_path, cropped_resized_image)
                print(f"Processed and saved {file_name}")
            else:
                print(f"Failed to read {file_name}")
