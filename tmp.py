import pickle
import cv2
import numpy as np

with open(r'tmp.obj', 'rb') as output_file:
    d = pickle.load(output_file)


xyz_2 = np.array(d['xyz_2'], dtype=np.float64)
uv_left_1 = np.array(d['uv_left_1'], dtype=np.float64)
intrinsics = np.array(d['intrinsics'], dtype=np.float64)

_, rvec, tvec, is_inlier = cv2.solvePnPRansac(objectPoints=xyz_2, imagePoints=uv_left_1, cameraMatrix=intrinsics[:, :3], distCoeffs=None)

print('Done!')