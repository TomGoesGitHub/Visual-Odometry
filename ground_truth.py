# visualize the driving 
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def draw_coordinate_frame_2d(ax, R,t):
    '''Projects the 3d coorindate frame represented by the rotation-matrix R
    and the translation-vector t into the 2d plane'''
    x_in_world, _, z_in_world = t 
    ihat_in_world, _, khat_in_world = R.T
    # note: the gps sensor is mounted s.t. the earth surface is repesented by the xz-plane, therefore we need 
    #       to project into this plane
    
    quiv_x = np.repeat(x_in_world, repeats=2, axis=0)
    quiv_y = np.repeat(z_in_world, repeats=2, axis=0)
    quiv_u = np.array([ihat_in_world[0], khat_in_world[0]])
    quiv_v = np.array([ihat_in_world[2], khat_in_world[2]])
    quiv = ax.quiver(quiv_x, quiv_y, quiv_u, quiv_v,
                     scale=15, scale_units='width', color=['red', 'blue'])
    return quiv

df = pd.read_csv('data_odometry_poses\\dataset\\poses\\00.txt', sep=' ')
times = pd.read_csv('dataset\\sequences\\00\\times.txt', header=None)
dir = 'dataset\\sequences\\00'

fig, axes = plt.subplot_mosaic(mosaic='BC\nAC', figsize=(20,10))
freq = 5

# ground truth
x = df.iloc[:,3]
y = df.iloc[:,11]
axes['C'].scatter(x,y,c='grey')

for i in range(len(times)):
    if i % freq == 0:    
        # camera view
        for ax, subdir in zip([axes['A'], axes['B']], ['image_0', 'image_1']):
            img_path = os.path.join(dir, subdir, '%06d.png'%i)
            img = plt.imread(img_path)
            ax.imshow(img, cmap='gray')

        # current position
        Rt = df.iloc[i].to_numpy().reshape(3,4)
        current_pos = axes['C'].scatter(x[i], y[i], c='black')
        R, t = Rt[:,:3], Rt[:,3]
        frame_gt = draw_coordinate_frame_2d(axes['C'], R, t)

        # pause and clean-up
        plt.pause(0.01)
        current_pos.remove()
        frame_gt.remove()
        axes['A'].clear()
        axes['B'].clear()

plt.show()