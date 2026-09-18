# show all seismic 3D data
import numpy as np
import os
from seismic_canvas import (SeismicCanvas, volume_slices, XYZAxis, Colorbar)
import matplotlib.pyplot as plt
import numpy as np

os.chdir("C:/Users/20649/Seismic/seis_diffusion_SR/seis_diffusion_SR/dataset/big_pieces_recombined/test-axis/") # 设置工作目录
file_chdir = os.getcwd() # 获得工作目录

def data_vision(path, volume, name):

    
    print(volume.shape)
    if volume.shape == (1, 128, 128, 96):
        volume = np.squeeze(volume)
        print(volume.shape)
    volume = volume.transpose(1, 2, 0) # 120 012 021 201 210 102
    axis_scales = (1, 1, 1) # isotropoic axes

    visual_nodes = volume_slices(volume,
        x_pos=20, y_pos=20, z_pos=13,
        seismic_coord_system=True)
    xyz_axis = XYZAxis(
        loc=(84.0, 77.0),
        seismic_coord_system=True)
    colorbar = Colorbar(cmap='seismic', clim=(volume.min(), volume.max()),
                        label_str='Amplitude', label_size=8, tick_size=6)

    title = '{}/{}'.format(path, name)

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
                            scale_factor=357.07137860968675,
                            center = [128, 48, 128],
                            fov = 30,
                            elevation = 21.5,
                            azimuth = 152.0,
                            zoom_factor = 1.2 # >1: zoom in; <1: zoom out
                            )

    canvas.measure_fps()
    canvas.app.run()


for root, dirs, files in os.walk(file_chdir): # os.walk会便利该目录下的所有文件
    for file in files:
        if os.path.splitext(file)[-1] == '.npy': # 判断文件格式是否符合npy格式
            path = root + '/' + file
            file_npy = np.load(path)
            print(path)
            data_vision(root, file_npy, os.path.splitext(file)[0])

   
