import numpy as np
from scipy.spatial.transform import Rotation

def homogenious_matrix(R=np.eye(3), t=np.zeros(3)):
    R, t = np.array(R), np.array(t)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,-1] = t
    return T

def normalize_quaternion(q):
    return q / np.linalg.norm(q)

def normalize_rotation_matrix(R):
    # note: it is easier to normalize the quaternions than the rotation matrix,
    #       therefore transform into quaternions first
    q = Rotation.from_matrix(R).as_quat()
    q_normalized = normalize_quaternion(q)
    R_normalized = Rotation.from_quat(q_normalized).as_matrix()
    return R_normalized