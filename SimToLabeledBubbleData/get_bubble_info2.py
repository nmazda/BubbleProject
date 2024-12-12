import os
import numpy as np
from mayavi import mlab
from tvtk.util.ctf import ColorTransferFunction
from scipy.ndimage import label
import argparse
import json
import warnings
import cv2  # Import OpenCV

warnings.filterwarnings("ignore")  

# Dimensions of the data
x_dim = 80
y_dim = 80
z_dim = 1000
chunk_size = 80 

# Read the binary .dat file
def read_dat_file(file_path):
    try:
        data = np.fromfile(file_path, dtype=np.uint32)
        data = data.reshape((z_dim, y_dim, x_dim))
    except Exception as e:
        print(f"Error reading or reshaping the file {file_path}: {e}")
        return None
    return data

# Function to preprocess data and create a binary mask
def preprocess_data(data, lower_threshold, upper_threshold):
    binary_mask = np.zeros_like(data, dtype=np.uint32)
    binary_mask[(data > lower_threshold) & (data < upper_threshold)] = 1 
    return binary_mask

# Function to detect bubbles and draw bounding boxes
def detect_and_draw_bubbles(data):
    labeled_array, num_features = label(data)
    bubble_info = []

    for bubble_label in range(0, num_features + 1):
        bubble_size = int(np.sum(labeled_array == bubble_label))
        bubble_positions = np.where(labeled_array == bubble_label)
        min_x, min_y, min_z = [int(np.min(pos)) for pos in bubble_positions]
        max_x, max_y, max_z = [int(np.max(pos)) for pos in bubble_positions]
        outline = (min_x, max_x, min_y, max_y, min_z, max_z)
        bubble_info.append((bubble_label, bubble_size, outline))

    bubble_info.sort(key=lambda x: x[1], reverse=True)
    bubble_info = bubble_info[1:]  

    for bubble_label, _, outline in bubble_info:
        bubble_positions = np.where(labeled_array == bubble_label)
        min_x, min_y, min_z = [int(np.min(pos)) for pos in bubble_positions]
        max_x, max_y, max_z = [int(np.max(pos)) for pos in bubble_positions]

        x, y, z = np.mgrid[min_x:max_x + 1:2j, min_y:max_y + 1:2j, min_z:max_z + 1:2j]
        scalars = np.zeros_like(x)
        mlab.pipeline.surface(mlab.pipeline.scalar_field(x, y, z, scalars), color=(0, 1, 0), opacity=0.0)

    print(f"Number of bubbles detected: {len(bubble_info)}")
    return bubble_info

# Function to detect bubble borders using OpenCV
def detect_bubble_borders(image_path):
    # Read the saved image
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply a binary threshold to the image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bubble_borders = []
    for contour in contours:
        # Get the bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)
        bubble_borders.append((x, y, w, h))  # Store x, y, width, height
    
    return bubble_borders

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bubble Detection Script')
    parser.add_argument('--input_dir', type=str, default='SimToLabeledBubbleData/VOFdata', help='Input directory containing .dat files')
    parser.add_argument('--output_dir', type=str, default='SimToLabeledBubbleData/Uncropped', help='Output directory for saving results')
    parser.add_argument('--json_output_dir', type=str, default='SimToLabeledBubbleData/bubble_loc_data', help='Output directory for saving JSON files')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(args.json_output_dir):
        os.makedirs(args.json_output_dir)

    fig = mlab.figure(bgcolor=(1, 1, 1))

    for file_name in os.listdir(args.input_dir):
        if file_name.endswith('.dat'):
            file_path = os.path.join(args.input_dir, file_name)
            data = read_dat_file(file_path)
            if data is not None:
                print(f"Processing file: {file_name}")
                data = np.transpose(data, (2, 1, 0))

                views = ['xz', 'yz']
                num_chunks = z_dim // chunk_size

                bubble_info_dict = {}

                for view in views:
                    for chunk_idx in range(num_chunks):
                        start_z = z_dim - (chunk_idx + 1) * chunk_size
                        end_z = z_dim - chunk_idx * chunk_size
                        chunk_data = data[:, :, start_z:end_z]

                        lower_threshold = 0.5 * 1000000000  
                        upper_threshold = 1.05 * 1000000000  
                        binary_mask = preprocess_data(chunk_data, lower_threshold, upper_threshold)

                        mlab.clf()

                        src = mlab.pipeline.scalar_field(chunk_data)
                        vol = mlab.pipeline.volume(src, vmin=chunk_data.min(), vmax=chunk_data.max())

                        vol._otf.remove_all_points()
                        th = 1.05 * 1000000000
                        vol._otf.add_point(th, 1.0)
                        vol._otf.add_point(th + 1, 0.0)

                        ctf = ColorTransferFunction()
                        ctf.add_rgb_point(chunk_data.min(), 0.0, 0.0, 0.0)
                        vol._volume_property.set_color(ctf)

                        if view == 'xz':
                            vol.scene.camera.position = [x_dim / 2, 50 * y_dim / 2, z_dim / 2]
                            vol.scene.camera.focal_point = [x_dim / 2, y_dim / 2, z_dim / 2]
                            vol.scene.camera.view_up = [0, 1, 0]
                        elif view == 'yz':
                            vol.scene.camera.position = [50 * x_dim / 2, y_dim / 2, z_dim / 2]
                            vol.scene.camera.focal_point = [x_dim / 2, y_dim / 2, z_dim / 2]
                            vol.scene.camera.view_up = [0, 0, 1]

                        vol.scene.camera.parallel_projection = True

                        bubbles = detect_and_draw_bubbles(binary_mask)

                        chunk_info_key = f"{file_name.replace('.dat','')}_{chunk_idx}_{view}"
                        bubble_info_dict[chunk_info_key] = bubbles

                        save_path = os.path.join(args.output_dir, os.path.splitext(file_name)[0] + '_' + str(chunk_idx) + '_' + view + '.png')
                        mlab.savefig(save_path)
                        print(f"Image saved as {save_path}")

                        # Detect bubble borders
                        bubble_borders = detect_bubble_borders(save_path)
                        print(f"Detected borders for {save_path}: {bubble_borders}")

                        # Add bubble border information to bubble_info_dict
                        for idx, bubble in enumerate(bubbles):
                            bubble_info_dict[chunk_info_key][idx] = (*bubble, bubble_borders[idx] if idx < len(bubble_borders) else None)

                # Convert all numpy.int64 types to int
                for key in bubble_info_dict:
                    print(key)
                    bubble_info_dict[key] = [(int(b[0]), int(b[1]), tuple(map(int, b[2])), b[3]) for b in bubble_info_dict[key]]

                json_save_path = os.path.join(args.json_output_dir, os.path.splitext(file_name)[0] + '_bubble_info.json')
                with open(json_save_path, 'w') as json_file:
                    json.dump(bubble_info_dict, json_file, indent=4)
                print(f"Bubble information saved as {json_save_path}")
            else:
                print(f"Data loading failed for file: {file_name}")

    mlab.close(fig)
