import os
import argparse
import numpy as np
from mayavi import mlab
from tvtk.util.ctf import ColorTransferFunction

# Dimensions of the data
x_dim = 80
y_dim = 80
z_dim = 1000

# Read the binary .dat file
def read_dat_file(file_path):
    try:
        data = np.fromfile(file_path, dtype=np.uint32)
        data = data.reshape((z_dim, y_dim, x_dim))
    except Exception as e:
        print(f"Error reading or reshaping the file {file_path}: {e}")
        return None
    return data


def main(input_dir):
    # Process each file in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.dat'):
            file_path = os.path.join(input_dir, file_name)
            data = read_dat_file(file_path)
            if data is not None:
                print(f"Processing file: {file_name}")
                data = np.transpose(data, (2, 1, 0))

                fig = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))
                src = mlab.pipeline.scalar_field(data)
                vol = mlab.pipeline.volume(src, vmin=data.min(), vmax=data.max())

                vol._volume_property.scalar_opacity_unit_distance = 10.0

                # Adjust opacity transfer function for transparency
                vol._otf.add_point(data.min(), 0.0)
                vol._otf.add_point(data.min() + 0.1 * (data.max() - data.min()), 0.0)
                vol._otf.add_point(data.max() - 0.1 * (data.max() - data.min()), 1.0)
                vol._otf.add_point(data.max(), 0.0)

                # Define a custom colormap (black and white)
                ctf = ColorTransferFunction()
                ctf.add_rgb_point(data.min(), 0.0, 0.0, 0.0)
                ctf.add_rgb_point(data.min() + 0.1 * (data.max() - data.min()), 0.0, 0.0, 0.0)
                ctf.add_rgb_point(data.max() - 0.1 * (data.max() - data.min()), 0.0, 1.0, 0.0)
                ctf.add_rgb_point(data.max(), 0.0, 0.0, 0.0)

                vol._volume_property.set_color(ctf)

                vol.scene.camera.position = [50 * x_dim / 2, y_dim / 2, z_dim / 2]
                vol.scene.camera.focal_point = [x_dim / 2, y_dim / 2, z_dim / 2]
                vol.scene.camera.view_up = [0, 0, 1]

                # Enable parallel projection
                vol.scene.camera.parallel_projection = True

                mlab.outline(color=(0, 0, 0), extent=(0, x_dim - 1, 0, y_dim - 1, 0, z_dim - 1))

                axes = mlab.axes(x_axis_visibility=True, y_axis_visibility=True, z_axis_visibility=True, color=(0, 0, 0))
                axes.label_text_property.font_size = 2
                axes.label_text_property.bold = False

                # Show the visualized data without saving the image
                mlab.show()

            else:
                print(f"Data loading failed for file: {file_name}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Visualize simulation data.')
    parser.add_argument('--input_dir', required=True, help='Path to the input directory containing .dat files')

    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.input_dir)
