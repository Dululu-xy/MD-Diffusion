from seismic_canvas import (SeismicCanvas, volume_slices, XYZAxis, Colorbar)
import matplotlib.pyplot as plt
import torch
import numpy as np

# import pyglet

# # ... Your existing code ...

# # Create a Pyglet window
# window = canvas.app.window

# # Define a function to capture the current frame and save it as an image
# def save_and_close(dt):
#     screenshot = pyglet.image.get_buffer_manager().get_color_buffer().get_texture()
#     screenshot.save('output_image.png')  # Replace 'output_image.png' with your desired file name and format
#     window.close()

# # Schedule the function to be called once (you can change this to 'pyglet.clock.schedule_interval' for periodic captures)
# pyglet.clock.schedule_once(save_and_close, 1.0)  # Adjust the delay (1.0 seconds in this example) as needed

# # Start the Pyglet event loop
# pyglet.app.run()

def data_vision(tensor_data, name, min_max=(-1, 1)):

    for i in range(tensor_data.shape[0]):
        volume = tensor_data.squeeze().float().cpu().clamp_(*min_max).numpy()[i]
        print(volume.shape)
        volume = volume.transpose(1, 2, 0)
        axis_scales = (1, 1, 1) # isotropoic axes

        visual_nodes = volume_slices(volume,
            x_pos=95, y_pos=127, z_pos=0,
            seismic_coord_system=False)
        xyz_axis = XYZAxis(loc=(84.0, 77.0),seismic_coord_system=False)
        colorbar = Colorbar(cmap='grays', clim=(volume.min(), volume.max()),
                            label_str='Amplitude', label_size=8, tick_size=6)

        title = name + '3D_vision_{}'.format(i)

        # Run the canvas.
        canvas = SeismicCanvas(title=title,
                                visual_nodes=visual_nodes,
                                xyz_axis=xyz_axis,
                                colorbar=colorbar,
                                # Set the option below=0 will hide the colorbar region
                                # colorbar_region_ratio=0,
                                axis_scales=axis_scales,
                                # Manual camera setting below.
                                # auto_range=False,
                                scale_factor=105.67819327246913,
                                center = [64, 64, 64],
                                fov = 30,
                                elevation = 19.5,
                                azimuth = -58.5,
                                zoom_factor = 1.2 # >1: zoom in; <1: zoom out
                                )

        canvas.measure_fps()
        canvas.app.run()


'''
128*128
===== All useful parameters ====
Canvas size = (800, 720)
Camera:
 - scale_factor = 211.35638654493826
 - center = [64.0, 64.0, 64.0]
 - fov = 30.0
 - elevation = 19.5
 - azimuth = -58.5
 - roll = 0.0
 - zoom factor = 1.2
Slices:
 - x: [127]
 - y: [127]
 - z: [0]
XYZAxis loc = (84.0, 77.0)'''

'''
64*64
===== All useful parameters ====
Canvas size = (800, 720)
Camera:
 - scale_factor = 105.67819327246913
 - center = [32.0, 32.0, 32.0]
 - fov = 30.0
 - elevation = 19.5
 - azimuth = -58.5
 - roll = 0.0
 - zoom factor = 1.2
Slices:
 - x: [49]
 - y: [63]
 - z: [0]
XYZAxis loc = (84.0, 77.0)'''
