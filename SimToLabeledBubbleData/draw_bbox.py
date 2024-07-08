import os
import json
import cv2
import argparse

def draw_bounding_boxes(image, bboxes, view, color=(0, 255, 0)):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for bbox in bboxes:
        _, _, (min1, max1, min_z, max_z, _, _) = bbox

        if view == 'xz':
            # Draw xz view with reversed y and x axes
            start_point = (new_size[1]- int(min1), int(new_size[1] - max_z))
            end_point = (new_size[1]- int(max1), int(new_size[1] - min_z))
        elif view == 'yz':
            # Draw yz view with reversed y axis
            start_point = (int(max1), new_size[1] - int(min_z))
            end_point = (int(min1), new_size[1] - int(max_z))

        image = cv2.rectangle(image, start_point, end_point, color, 2)

    return image

def adjust_bounding_boxes(bboxes, original_size, new_size, view):
    orig_width, orig_height = original_size
    new_width, new_height = new_size

    scale1 = (new_width / orig_width)
    scale_z = (new_height / orig_height)

    adjusted_bboxes = []
    for bbox in bboxes:
        label, size, (min_x, max_x, min_y, max_y, min_z, max_z) = bbox

        if view == 'xz':
            min1 = min_x * scale1
            max1 = max_x * scale1
            min_z = min_z * scale_z
            max_z = max_z * scale_z
            adjusted_bboxes.append((label, size, (min1, max1, min_z, max_z, 0, 0)))
        elif view == 'yz':
            min2 = min_y * scale1
            max2 = max_y * scale1
            min_z = min_z * scale_z
            max_z = max_z * scale_z
            adjusted_bboxes.append((label, size, (min2, max2, min_z, max_z, 0, 0)))

    return adjusted_bboxes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw bounding boxes on images using JSON data')
    parser.add_argument('--input_dir', type=str, default='SimToLabeledBubbleData/Real', help='Input directory containing images')
    parser.add_argument('--output_dir', type=str, default='SimToLabeledBubbleData/LabelledBubbleData', help='Output directory to save processed images')
    parser.add_argument('--json_dir', type=str, default='SimToLabeledBubbleData/bubble_loc_data', help='Directory containing JSON files')
    parser.add_argument('--original_size', type=int, nargs=2, default=(80, 80), help='Original size (width, height) of the images')
    parser.add_argument('--new_size', type=int, nargs=2, default=(256, 256), help='New size (width, height) for the resized images')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    json_dir = args.json_dir
    original_size = tuple(args.original_size)
    new_size = tuple(args.new_size)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.png'):
            file_path = os.path.join(input_dir, file_name)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            base_parts = file_name.split('_')
            json_file_name = '_'.join(base_parts[:3]) + '_bubble_info.json'
            json_path = os.path.join(json_dir, json_file_name)

            view = base_parts[-1].split('.')[0]

            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    bubble_info_dict = json.load(f)

                json_key = os.path.splitext(file_name)[0]

                if json_key in bubble_info_dict:
                    bboxes = bubble_info_dict[json_key]
                    adjusted_bboxes = adjust_bounding_boxes(bboxes, original_size, new_size, view)
                    image_with_boxes = draw_bounding_boxes(image, adjusted_bboxes, view)

                    save_path = os.path.join(output_dir, file_name)
                    cv2.imwrite(save_path, image_with_boxes)
                    print(f"Image with bounding boxes saved as {save_path}")
                else:
                    print(f"No bounding box data found for {file_name}")
            else:
                print(f"JSON file not found for {file_name}")
