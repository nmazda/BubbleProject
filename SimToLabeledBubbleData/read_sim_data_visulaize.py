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
        print(f"Error reading or reshaping the file: {e}")
        return None
    return data

file_path = 'VOFdata/fl_0_0315.dat'
data = read_dat_file(file_path)
if data is not None:
    print(f"Data shape: {data.shape}")
    data = np.transpose(data, (2, 1, 0))

    fig = mlab.figure(size=(1000, 1000))
    src = mlab.pipeline.scalar_field(data)
    vol = mlab.pipeline.volume(src, vmin=data.min(), vmax=data.max())

    vol._volume_property.scalar_opacity_unit_distance = 10.0

    # Adjust opacity transfer function for transparency
    vol._otf.add_point(data.min(), 0.0)  
    vol._otf.add_point(data.min() + 0.1 * (data.max() - data.min()), 0.0) 
    vol._otf.add_point(data.max() - 0.1 * (data.max() - data.min()), 1.0)  
    vol._otf.add_point(data.max(), 0.0) 

    # Define a custom blue colormap
    ctf = ColorTransferFunction()
    ctf.add_rgb_point(data.min(), 0.0, 0.0, 0.0)  
    ctf.add_rgb_point(data.min() + 0.1 * (data.max() - data.min()), 0.0, 0.0, 0.0) 
    ctf.add_rgb_point(data.max() - 0.1 * (data.max() - data.min()), 0.0, 0.0, 0.0) 
    ctf.add_rgb_point(data.max(), 0.0, 0.0, 0.0)  

    vol._volume_property.set_color(ctf)

    vol.scene.camera.position = [x_dim / 2, y_dim / 2, 3 * z_dim]  
    vol.scene.camera.focal_point = [x_dim / 2, y_dim / 2, z_dim / 2]  
    vol.scene.camera.view_up = [1, 0, 0] 

    mlab.outline(color=(0, 0, 0), extent=(0, x_dim-1, 0, y_dim-1, 0, z_dim-1))
        
    axes = mlab.axes(x_axis_visibility=True, y_axis_visibility=True, z_axis_visibility=True, color=(0, 0, 0))
    axes.label_text_property.font_size = 2 
    axes.label_text_property.bold = False 

    mlab.show()
else:
    print("Data loading failed. Exiting.")
