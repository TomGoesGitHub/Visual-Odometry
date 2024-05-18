import os
import csv

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

from utils.spatial import homogenious_matrix, normalize_rotation_matrix
from utils.visualization import draw_coordinate_frame_2d, draw_matches_custom
from visual_odometry import FeatureBasedVisualOdometryCallback
from io_kittislam import KittiSlamDataloader


class SimulationCallback(FeatureBasedVisualOdometryCallback):
    '''A Callback suited for the used Simulation. The callback is used to extract information during the
    computation (which would be private otherwise), such that the internal state of the Visual-Odometry
    can be visualized. All data required by the simulation class is collected by this callback.'''
    def __init__(self, simulation):
        super().__init__()
        self.simulation = simulation
    
    def after_stereo_matching(self, locals):
        '''At runtime, after stereo-matching in the VisualOdometry class this callback is triggered.
        It projects the identified keypoints onto the scene-images and highlights the found matches.

        Args:
            locals: private locals from the visual odometry class (important values will be extracted)
        '''        
        if self.simulation.render:
            self.simulation.axes['D'].clear()
            self.simulation.axes['D'].set_title('Stereo Matching')
             
            img_matches = draw_matches_custom(
                img1=cv2.cvtColor(locals['img_l'], cv2.COLOR_GRAY2RGB),
                kp1=locals['kp_l'],
                img2=cv2.cvtColor(locals['img_r'], cv2.COLOR_GRAY2RGB),
                kp2=locals['kp_r'],
                matches=locals['matches']
            )
            self.simulation.axes['D'].imshow(img_matches)
            self.simulation.axes['D'].tick_params(which='both',
                                                  bottom=False, left=False,
                                                  labelbottom=False, labelleft=False) 

    def after_triangulation(self, locals):
        pass # do nothing
    
    def after_track_matching(self, locals):
        '''At runtime, after track-matching in the VisualOdometry class this callback is triggered.
        It 1) creates a disparity map and 2) maps the optical flow between two sucessive frames onto
        the image plane.

        Args:
            locals: private locals from the visual odometry class (important values will be extracted)
        '''   
        # unpack
        img_l1, img_r1 = locals['img_pair_1']
        img_l2, img_r2 = locals['img_pair_2']
        uv_l1, uv_l2 = locals['uv_left_1'], locals['uv_left_2']
        
        # disparity map
        ax = self.simulation.axes['F']
        ax.clear(), ax.set_title('Disparity Map')
        disparity = np.zeros(shape=img_l1.shape[:2])
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=7)
        disparity = stereo.compute(img_l1, img_r1)
        ax.imshow(disparity,'viridis')
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        
        # optical flow
        blended_l = cv2.addWeighted(img_l1, 0.5, img_l2, 0.5, 0.0)
        blended_l = cv2.cvtColor(blended_l, cv2.COLOR_GRAY2RGB)
        
        for pt1, pt2 in zip(uv_l1, uv_l2):
            cv2.line(blended_l, pt1, pt2, color=(0,0,255), thickness=1)

        ax = self.simulation.axes['B']
        ax.clear(), ax.set_title('Optical Flow (Left Image only)')
        ax.imshow(blended_l)
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)        

class Simulation():
    '''Simulation class for the Visual Odometry Problem applied to the Kitti-Slam-Dataset.
    Runs the simulation by processing the visual input data and benchmarking against the ground truth.
    Important aspects of the simulation are visualized in a video-like fashion in real time'''
    def __init__(self, dir, sequence, visual_odometry, frame_usage_freq=1, logging_filepath=None, render=True):
        '''
        Args:
            dir (str): data directory, which contains all Kitti-Slam-sequences
            sequence (int): number of sequence (e.g. sequence=0 corresponds to sub-directory "00") 
            visual_odometry: instance of VO-class
            frame_usage_freq (int): use every n-th frame
            logging_filepath (str, optional): Location (csv-file), where results are logged to.
                If None, no logging is performed. Defaults to None.
            render (bool, optional): Wether to render the experiment in a video-like fashion. Defaults to True.
        '''        
        # data
        img_stream_dir = os.path.join(dir, 'sequences', '%02d'%sequence)
        self.img_stream = KittiSlamDataloader(dir=img_stream_dir)
        
        gt_dir = os.path.join(dir, 'poses', '%02d.txt'%sequence)
        self.df_ground_truth = pd.read_csv(gt_dir, sep=' ', header=0)
        self.t_max = len(self.df_ground_truth) - 1
        self.logging_filepath = logging_filepath

        # visualization
        self.render = render
        self.frame_usage_freq = frame_usage_freq
        self.fig, self.axes = plt.subplot_mosaic(mosaic='AADD\nCCFF\nCCBB', figsize=(20,10),)

        # odometry model
        self.odom = visual_odometry
        if self.render:
            self.odom.callback = SimulationCallback(simulation=self)

        # state, to be updated
        self.t = self.frame_usage_freq
        assert self.t>0 # note: >0, because we need 2 image pairs for tracking
        Rt_gt_current = self.df_ground_truth.iloc[self.t].to_numpy().reshape(3,4)
        self._T_world_to_cam = np.eye(4)
        self._T_world_to_cam[:3, :] = Rt_gt_current

    @property
    def T_world_to_cam(self):
        return self._T_world_to_cam
    
    @T_world_to_cam.setter
    def T_world_to_cam(self, value):
        R = value[:3, :3]
        t = value[:3, 3]
        # renormalize, in order to avoid numerical drift
        R_normalized = normalize_rotation_matrix(R)
        self._T_world_to_cam = homogenious_matrix(R_normalized,t)

    def visualize(self, current_img_pair):
        '''Visualizes those parts of the current state, which this class has direct acess to.
        Other parts are visualized by the callback-mechanism.

        Args:
            current_img_pair: image-tuple (left and right frame)
        '''        
        # state estimation
        R, t = self.T_world_to_cam[:3, :3], self.T_world_to_cam[:3, 3]
        x, z = t[0], t[2]
        self.axes['C'].scatter(x, z, c='green', s=3)
        frame_estimated, pos_estimated = draw_coordinate_frame_2d(self.axes['C'], R,t, color='green', alpha=1)

        # ground truth
        Rt_gt = self.df_ground_truth.iloc[self.t].to_numpy().reshape(3,4)
        R_gt, t_gt = Rt_gt[:,:3], Rt_gt[:,3]
        frame_gt, pos_gt = draw_coordinate_frame_2d(self.axes['C'], R_gt, t_gt, color='black', alpha=0.5)
        self.axes['C'].set_title('Map (Estimation vs. Groundtruth)')

        # camera view
        img_pair_stacked = np.concatenate(current_img_pair, axis=1)
        self.axes['A'].clear()
        self.axes['A'].imshow(img_pair_stacked, cmap='gray')
        self.axes['A'].axis('off')
        self.axes['A'].set_title('Stereo Vision (Input)')

        # pause and clean-up
        plt.tight_layout()
        plt.pause(0.5)
        
        pos_gt.remove(), frame_gt.remove()
        frame_estimated.remove(), pos_estimated.remove()

    def write_logging(self):
        '''Write to the specified csv-file.'''
        already_exists = os.path.exists(self.logging_filepath)
        with open(self.logging_filepath, mode='a', newline='\n') as csvfile:
            row = {'t': self.t,
                   'x': self.T_world_to_cam[0, -1],
                   'z': self.T_world_to_cam[2, -1],}
            # note: for now, we only logg the position for simplicity
            writer = csv.DictWriter(csvfile, fieldnames=row.keys()) 
            if not already_exists:
                writer.writeheader()
            writer.writerow(row) 
    

    def step(self):
        '''Steps the simulation forward by one timetep.'''        
        # visual odometry
        prior_img_pair = self.img_stream[self.t-self.frame_usage_freq]
        current_img_pair = self.img_stream[self.t]
        T_prior_to_current = self.odom(prior_img_pair, current_img_pair)

        self.T_world_to_cam = self.T_world_to_cam @ T_prior_to_current
        
        if self.render:
            self.visualize(current_img_pair)
        
        if self.logging_filepath:
            self.write_logging()

        self.t += self.frame_usage_freq

    def run(self):
        '''Runs the simulation.'''
        if self.render:
            plt.get_current_fig_manager().window.showMaximized()
            x_gt, z_gt = self.df_ground_truth.iloc[:, 3], self.df_ground_truth.iloc[:, 11]
            self.axes['C'].scatter(x_gt, z_gt, c='grey')
        
        initial_t = self.t
        progress_bar = tqdm.tqdm(total=self.t_max - self.t, initial=initial_t)
        while self.t < self.t_max:
            progress_bar.update(self.frame_usage_freq)
            self.step()
        progress_bar.close()