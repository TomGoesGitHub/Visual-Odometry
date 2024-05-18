import os

import cv2

from visual_odometry import StereoCamera, FeatureBasedVisualOdemetry
from simulation import Simulation
from config import local_data_dir
# todo/note: in order to run the program you need to change the local data directory in the config-file
from sensor_setup import sensor_setup

def main():
    # camera model (intrinsics)
    f_u = sensor_setup.T_cam1_to_img1[0,0]
    f_v = sensor_setup.T_cam1_to_img1[1,1]
    c_u = sensor_setup.T_cam1_to_img1[0,2]
    c_v = sensor_setup.T_cam1_to_img1[1,2]
    baseline = - 1/f_u * sensor_setup.T_cam1_to_img1[0,3]
    camera_model = StereoCamera(baseline, focal_len=(f_u, f_v), center_pos=(c_u, c_v))

    # visual odometry setup
    odometry = FeatureBasedVisualOdemetry(
        detector=cv2.SIFT_create(),
        descriptor=cv2.SIFT_create(),
        matcher=cv2.FlannBasedMatcher(indexParams=dict(algorithm=1, trees=5),
                                      searchParams=dict(checks=50)),
        camera_model=camera_model
    ) 

    # simulation
    sequence=0
    logging_file = os.path.join('results', f'logging_for_sequence_{"%02d"%sequence}.csv')
    simulation = Simulation(local_data_dir, sequence, visual_odometry=odometry, frame_usage_freq=5,
                            logging_filepath=logging_file, render=True)
    simulation.run()
    print('Done!')

if __name__ == '__main__':
    main()



