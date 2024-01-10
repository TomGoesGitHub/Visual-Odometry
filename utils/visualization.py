import numpy as np
import matplotlib.pyplot as plt
import cv2

def draw_coordinate_frame_2d(ax, R,t, color='black', alpha=1):
    '''Projects the 3d coorindate frame represented by the rotation-matrix R
    and the translation-vector t into the 2d plane'''
    x_in_world, _, z_in_world = t 
    ihat_in_world, _, khat_in_world = R.T
    # note: the gps sensor is mounted s.t. the earth surface is repesented by the xz-plane, 
    # therefore we need to project into this plane
    
    quiv_x = np.repeat(x_in_world, repeats=2, axis=0)
    quiv_y = np.repeat(z_in_world, repeats=2, axis=0)
    quiv_u = np.array([ihat_in_world[0], khat_in_world[0]])
    quiv_v = np.array([ihat_in_world[2], khat_in_world[2]])
    quiv = ax.quiver(quiv_x, quiv_y, quiv_u, quiv_v,
                     scale=15, scale_units='width', color=['red', 'blue'], alpha=alpha)
    x_gt, z_gt = t[0], t[2]
    pos = ax.scatter(x_gt, z_gt, c=color)
    return quiv, pos

def draw_matches_custom(img1, kp1, img2, kp2, matches):
    blended = cv2.addWeighted(img1, 0.5, img2, 0.5, 0.0)
    cv2.drawKeypoints(blended, kp1, blended, color=(0,255,0))
    cv2.drawKeypoints(blended, kp2, blended, color=(255,0,0))
    for match in matches:
        pt1 = tuple(int(x) for x in kp1[match.queryIdx].pt)
        pt2 = tuple(int(x) for x in kp2[match.trainIdx].pt)
        blended = cv2.line(blended, pt1, pt2, color=(0,0,255), thickness=1)
    return blended # return ax.imshow(blended)

def draw_matches_3d(ax, xyz_0, xyz_1):
    ax.scatter(xyz_0[:, 0], xyz_0[:, -1], color='blue', s=1)
    ax.scatter(xyz_1[:, 0], xyz_1[:, -1], color='red', s=1)
    n = len(xyz_0)
    for i in range(n):
        x = [xyz_0[i, 0], xyz_1[i, 0]]
        y = [xyz_0[i, 2], xyz_1[i, 2]]
        ax.plot(x, y, c='k', lw=1)


