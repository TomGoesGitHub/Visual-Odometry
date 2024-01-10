'''
information about the sensor setup can be found in
[Geiger, 2013: Vision Meets Robotics - The Kitti Dataset]
'''

import numpy as np
import pandas as pd
from utils.matrix import homogenious_matrix

class sensor_setup:
    # world coordinate system
    T_world_to_cam0_initial = homogenious_matrix(
        R=[[0,  0, 1],
           [0, -1, 0],
           [1,  0, 0]]
    )
    # note: initially, camera is positioned at (0,0,0) in world coordinates,
    #       but the cam-frame is rotated

    # extrinsics (aka rigid body transformations)
    T_gps_to_velo = homogenious_matrix(
        t=[0.81, -0.32, 0.8]
    )

    T_velo_to_cam0 = homogenious_matrix(
        R=[[0, -1,  0],
           [0,  0, -1],
           [1,  0,  0]],
        t=[0.27, -0.08, 0]
    )

    T_cam0_to_cam1 = homogenious_matrix(
        t=[0.54, 0, 0]
    )

    # intrinsics (rectification and projection in image-plane)
    T_cam_to_camRect = homogenious_matrix() # todo: is this even required?

    df = pd.read_csv('D:\\DATASETS\\Kitti_SLAM\\dataset\\sequences\\00\\calib.txt',
                    sep=' ', header=None, index_col=0)

    T_cam0_to_img0 = df.loc['P0:'].to_numpy().reshape(3,4)
    T_cam1_to_img1 = df.loc['P1:'].to_numpy().reshape(3,4)
    

if __name__ == '__main__':
    print(sensor_setup.T_cam0_to_img0)
    print(sensor_setup.T_cam1_to_img1)