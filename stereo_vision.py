import os

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.features import filter_horizontal_matches, filter_matches_by_distance_ratio, filter_best_matches
from utils.spatial import homogenious_matrix, normalize_rotation_matrix, kabsch_umeyama
from utils.visualization import draw_coordinate_frame_2d, draw_matches_custom, draw_matches_3d



class KittiSlamDataloader:
    # https://stackoverflow.com/questions/42983569/how-to-write-a-generator-class
    def __init__(self, dir):
        self.dir = dir
    
    def __getitem__(self, key):
        path_left = os.path.join(self.dir, 'image_0', '%06d.png'%key)
        path_right = os.path.join(self.dir, 'image_1', '%06d.png'%key)
        img_left, img_right = cv2.imread(path_left), cv2.imread(path_right)
        img_left, img_right = self.preprocess(img_left), self.preprocess(img_right)
        return img_left, img_right
    
    def preprocess(self, img):
        R, G, B = cv2.split(img)
        output1_R = cv2.equalizeHist(R)
        output1_G = cv2.equalizeHist(G)
        output1_B = cv2.equalizeHist(B)
        equ = cv2.merge((output1_R, output1_G, output1_B))
        return equ
        

class StereoCamera:
    def __init__(self, baseline, focal_len, center_pos):
        # intrinsics
        self.baseline = baseline # in meter
        self.focal_len = focal_len # in pixels
        self.center_pos = center_pos # in pixels, in pixel coordinates

    def intrincsic_matrix(self, which='left'):
        assert which in ['left', 'right']

        c_u, c_v, f_u, f_v = *self.center_pos, *self.focal_len 
        b = self.baseline if (which == 'right') else 0

        return np.array([[f_u, 0, c_u, -f_u*b],
                         [0, f_v, c_v,      0],
                         [0,   0,   1,      0]])
        
    
    def triangulate(self, uv_left, uv_right):
        '''
        Inverse camera model: Recover the depth from the stereo vision pixels.    

        see: First Principles of Computer Vision - Simple Stereo | Camera Calibration
        https://www.youtube.com/watch?v=hUVyDabn1Mg&t=330s
        '''        
        u_l, v_l = uv_left.T
        u_r, v_r = uv_right.T
        
        b, c_u, c_v, f_u, f_v = self.baseline, *self.center_pos, *self.focal_len # alias

        d = u_l - u_r # disparity
        x = b * (u_l - c_u) / d
        y = b * f_u/f_v * (v_l - c_v) / d
        z = b * f_u / d
        
        xyz = np.stack([x,y,z], axis=1)
        return xyz
    
    def forward(self, xyz):
        '''Forward camera model: Get pixel coordinates from 3D point.
        
        see: Geiger, 2013: Vision Meets Robotics - The Kitti Dataset, Eq. 4.
        '''
        b, c_u, c_v, f_u, f_v = self.baseline, *self.center_pos, *self.focal_len # alias
        
        T_xyz_to_img_left = np.array([[f_u, 0, c_u, 0],
                                      [0, f_v, c_v, 0],
                                      [0,   0,   1, 0]])
        
        T_xyz_to_img_right = np.array([[f_u, 0, c_u, -f_u*b],
                                       [0, f_v, c_v,      0],
                                       [0,   0,   1,      0]])

        xyz_homogenious = np.concatenate([xyz, np.ones([len(xyz),1])], axis=1).T
        z = xyz[:, 2]
        uv_homogenious_left = T_xyz_to_img_left @ xyz_homogenious/z
        uv_homogenious_right = T_xyz_to_img_right @ xyz_homogenious/z
        
        uv_left = uv_homogenious_left[:-1].T
        uv_right = uv_homogenious_right[:-1].T
        
        return uv_left.astype(np.int32), uv_right.astype(np.int32)

class FeatureBasedVisualOdometryCallback:
    '''Abstract Callback Class for Feature Based Visual Odometry.
    By subclassing, additional functionality can be added at the corresponding time.
    '''
    def after_stereo_matching(self, locals):
        pass
    
    def after_triangulation(self, locals):
        pass

    def after_track_matching(self, locals):
        pass
    

class FeatureBasedVisualOdemetry:
    def __init__(self, detector, descriptor, matcher, camera_model,
                 callback=FeatureBasedVisualOdometryCallback()):
        self.detector = detector
        self.descriptor = descriptor
        self.matcher = matcher
        self.camera_model = camera_model
        self.callback = callback

    def fit_spatial_transformation(self, xyz_1, xyz_2, n_ransac=1000):
        '''
        Estimating the Transformation between the Frames A and B based on 3d samples from frame A
        and B with RANSAC (random sample and consensus) and Point Cloud Alignment.

        Sources:
        - RANSAC: Fischler, M. and Bolles, R. (1981). Random sample and consensus: A paradigm for
                  model fitting with applications to image analysis and automated cartography. Comm.
                  of the ACM, 24(6):381-395. (ref. pages 12 and 127)
        - PC-Al.: S Umeyama, “Least-Squares Estimation of Transformation Parameters Between Two 
                  Point Patterns”, IEEE Transactions on Pattern Analysis and Machine Intelligence, 
                  13(4), 1991
        '''
        n_datapoints = len(xyz_1)
        xyz_1, xyz_2 = np.array(xyz_1), np.array(xyz_2)
        
        #fig_, ax_ = plt.subplots() # todo: tmp

        best_n_inliers, R, t = -1, np.eye(3), np.ones(3) # to be updated
        for _ in range(n_ransac):
            # ransac random sampling
            n_random_sampels = 10
            idx = np.arange(n_datapoints)
            train_idx = np.random.choice(idx, size=n_random_sampels, replace=False)
            xyz_train_1, xyz_train_2 = xyz_1[train_idx], xyz_2[train_idx] 

            # point cloud alignment
            R, tmp, t = kabsch_umeyama(
                A=xyz_train_1,
                B=xyz_train_2
            )

            # centroid_1 = np.mean(xyz_train_1, axis=0)
            # centroid_2 = np.mean(xyz_train_2, axis=0)
            # delta_1 = (xyz_train_1 - centroid_1)
            # delta_2 = (xyz_train_2 - centroid_2)
            # W = np.einsum('ij,ik->jk', delta_1, delta_2) # sum over outer products
            # #H = np.sum([d1.reshape([-1,1])@d2.reshape([-1,1]).T for d1, d2 in zip(delta_1, delta_2)], axis=0)
            # U,_,VT = np.linalg.svd(W)
            # V = VT.T
            # R = V @ np.diag([1,1, np.linalg.det(U)*np.linalg.det(V)]) @ U.T
            # t = -R @ centroid_1 + centroid_2
            
            # ransac rejection
            rejection_threshold = 0.1
            c = xyz_2 - (t + np.matmul(R, (xyz_1)[:, :, np.newaxis]).squeeze(-1))
            costs = np.einsum('ji,ik->i', c.T, c) # batch inner prod
            is_inlier = (costs <= rejection_threshold)
            n_inliers = sum(is_inlier)
            
            #ax_.hist(costs, bins=100, alpha=0.1, color='steelblue') # todo: tmp

            # update
            if n_inliers > best_n_inliers:
                #det = np.linalg.det(U)*np.linalg.det(V)
                best_is_inlier = is_inlier
                best_n_inliers = n_inliers
                T_a_to_b = homogenious_matrix(R.T, -R.T@t)
                #T_a_to_b = homogenious_matrix(R,t)
        
        #plt.show() # todo: tmp
        # T_a_to_b = homogenious_matrix(R,t)
        print('\n', best_n_inliers/len(is_inlier), np.linalg.det(R),'\n', tmp, '\n', np.round(T_a_to_b, decimals=2))
        return T_a_to_b, np.array(best_is_inlier)
    
    def match_stereo(self, img_l, img_r, kp_l, kp_r):
        ''' Pipeline for the computation and flitering of stereo-matches between 
        left and right image. The goal is to keep good matches only.
        See: Laganiere - Computer vision application programming cookbook, Chapter 9.'''        
        _, des_l = self.descriptor.compute(img_l, kp_l)
        _, des_r = self.descriptor.compute(img_r, kp_r)
        
        knn_matches = self.matcher.knnMatch(des_l, des_r, k=2)
        matches = filter_matches_by_distance_ratio(knn_matches)
        matches = filter_horizontal_matches(kp_l, kp_r, matches)
        # matches = filter_best_matches(matches, keep_n_best=750)
        self.callback.after_stereo_matching(locals()) # todo: maybe put at the end of this function

        # only keep matching keypoints
        kp_l = [kp_l[match.queryIdx] for match in matches]
        kp_r = [kp_r[match.trainIdx] for match in matches]
        return kp_l, kp_r
    
    def match_track(self, des_prior, des_current):
        ''' Pipeline for the computation and flitering of track-matches betweem two consecutive
        frames. The goal is to keep good matches only.
        See: Laganiere - Computer vision application programming cookbook, Chapter 9.'''
        knn_matches = self.matcher.knnMatch(des_prior, des_current, k=2)
        matches = filter_matches_by_distance_ratio(knn_matches)
        matches = filter_best_matches(matches, keep_n_best=750)
        return matches

    def run_stereo_vision_pipeline(self, img_left, img_right):
        '''Run Stereo Vision Pipeline: i.e. keypoint-extraction, keypoint-description, matching,
        and 3d reconstruction.'''
        # keypoint detection
        kp_left, kp_right = self.detector.detect(img_left), self.detector.detect(img_right)
        # kp_left = self.get_keypoints_from_tiles(img_left)
        # kp_right = self.get_keypoints_from_tiles(img_right)

        # stereo matching
        kp_left, kp_right = self.match_stereo(img_left, img_right, kp_left, kp_right)
        _, des_left = self.descriptor.compute(img_left, kp_left)
        _, des_right = self.descriptor.compute(img_right, kp_right)

        # pixel coordinates
        uv_left = cv2.KeyPoint.convert(kp_left).astype(int)
        uv_right = cv2.KeyPoint.convert(kp_right).astype(int)

        # triangulation
        xyz_cam = self.camera_model.triangulate(uv_left, uv_right)
        des = np.hstack([des_left, des_right])
        #self.callback.after_triangulation(locals())
        
        return xyz_cam, des, uv_left
    
    def __call__(self, img_pair_1, img_pair_2):
        '''Given an image pair for each camera position, calculate the best fit of
        the spatial transformation between the 2 camera positions.'''
        xyz_1, des1, uv_left_1 = self.run_stereo_vision_pipeline(*img_pair_1)
        xyz_2, des2, uv_left_2 = self.run_stereo_vision_pipeline(*img_pair_2)
        
        tracking_matches = self.match_track(des1, des2)
        idx_1 = np.array([match.queryIdx for match in tracking_matches])
        idx_2 = np.array([match.trainIdx for match in tracking_matches])
        xyz_1, xyz_2 = xyz_1[idx_1], xyz_2[idx_2]
        uv_left_1, uv_left_2 = uv_left_1[idx_1], uv_left_2[idx_2]
        self.callback.after_track_matching(locals())

        intrinsics = self.camera_model.intrincsic_matrix(which='left')
        
        _, rvec, tvec, is_inlier = cv2.solvePnPRansac(objectPoints=xyz_2,
                                                      imagePoints=uv_left_1.astype(float), # note: cv2 requires float here, which is kinda weird...
                                                      cameraMatrix=intrinsics[:, :3],
                                                      distCoeffs=None)
        rmat = cv2.Rodrigues(rvec)[0]    
        T_1_2 = homogenious_matrix(R=rmat, t=tvec)
        is_inlier = is_inlier.squeeze()
        
        #T_1_2, is_inlier= self.fit_spatial_transformation(xyz_1, xyz_2)

        xyz_1, xyz_2 = xyz_1[is_inlier], xyz_2[is_inlier]
        self.callback.after_triangulation(locals())
        return T_1_2

    def tile(self, img, nrows, ncols):
        height, width, _ = img.shape
        uu = np.linspace(0, width, ncols, dtype=np.int32)
        vv = np.linspace(0, height, nrows, dtype=np.int32)
        
        tiles, uv = [], []
        for i in range(len(uu)-1):
            u = uu[i]
            u_next = uu[i+1]
            for j in range(len(vv)-1):
                v = vv[j]
                v_next = vv[j+1]
    
                tile = img[v:v_next, u:u_next, :]
                tiles.append(tile)
                uv.append((u,v))

                # fig_ = plt.figure()
                # ax_ = fig_.add_subplot()
                # tile_in_zeros = np.zeros_like(img)
                # tile_in_zeros[v:v_next, u:u_next, :] = tile
                # ax_.imshow(tile_in_zeros)
                # plt.show()
        return tiles, uv

    def get_keypoints_from_tiles(self, img, nrows=10, ncols=30):
        tiles, uv = self.tile(img, nrows, ncols)
        kp_img = []
        
        for tile, (u,v) in zip(tiles, uv):
            # compute keypoints in tile
            kp_tile = self.detector.detect(tile) # [self.detector.detect(tile) for tile in tiles_l1]

            # correct coordinates (from tile pxiel coorinates to img pixel coordinates)
            for kp in kp_tile:
                kp.pt = (kp.pt[0]+u, kp.pt[1]+v)
            kp_img.extend(kp_tile)
        
        return kp_img        

    # def __call__(self, img_pair_1, img_pair_2):
    #     img_l1, img_r1 = img_pair_1
    #     img_l2, img_r2 = img_pair_2

    #     # keypoint detection
    #     # kp_l1 = self.get_keypoints_from_tiles(img_l1)
    #     # kp_r1 = self.get_keypoints_from_tiles(img_r1)
    #     kp_l1, kp_r1 = self.detector.detect(img_l1), self.detector.detect(img_r1)
         
    #     # stereo matching
    #     kp_l1, kp_r1 = self.match_stereo(img_l1, img_r1, kp_l1, kp_r1)

    #     # pixel coorindates
    #     uv_l1 = cv2.KeyPoint.convert(kp_l1)
    #     uv_r1 = cv2.KeyPoint.convert(kp_r1)

    #     # optical flow
    #     uv_l1, uv_r1, uv_l2, uv_r2 = self.optical_flow_stereo(img_pair_1, img_pair_2, uv_l1, uv_r1)
    #     self.callback.after_track_matching(locals())

    #     # triangulate
    #     xyz_1 = self.camera_model.triangulate(uv_l1, uv_r1)
    #     xyz_2 = self.camera_model.triangulate(uv_l2, uv_r2)
        
    #     # pose-estimation
    #     T_1_2, is_inlier = self.fit_spatial_transformation(xyz_1, xyz_2)
    #     xyz_1, xyz_2 = xyz_1[is_inlier], xyz_2[is_inlier]
    #     self.callback.after_triangulation(locals())
    #     return T_1_2

    def optical_flow_stereo(self, img_pair_1, img_pair_2, uv_l1, uv_r1):
        img_l1, img_r1 = img_pair_1
        img_l2, img_r2 = img_pair_2
        
        # optical flow left
        uv_l2, status_l, err_l = cv2.calcOpticalFlowPyrLK(prevImg=img_l1, nextImg=img_l2,
                                                          prevPts=uv_l1, nextPts=None)
        mask_success_l = np.asarray(status_l, dtype=bool).squeeze()
        
        # optical flow right
        uv_r2, status_r, err_r = cv2.calcOpticalFlowPyrLK(prevImg=img_r1, nextImg=img_r2,
                                                          prevPts=uv_r1, nextPts=None)
        mask_success_r = np.asarray(status_r, dtype=bool).squeeze()

        # only keep keypoints where the left and right flow was successful
        mask_success = np.logical_and(mask_success_l, mask_success_r)

        return uv_l1[mask_success], uv_r1[mask_success], uv_l2[mask_success], uv_r2[mask_success]
        

class SimulationCallback(FeatureBasedVisualOdometryCallback):
    def __init__(self, simulation):
        super().__init__()
        self.simulation = simulation
    
    def after_stereo_matching(self, locals):
        if self.simulation.trigger_visualization:
            self.simulation.axes['D'].clear()
            self.simulation.axes['D'].set_title('Stereo Matching')
             
            img_matches = draw_matches_custom(
                img1=locals['img_l'],
                kp1=locals['kp_l'],
                img2=locals['img_r'],
                kp2=locals['kp_r'],
                matches=locals['matches']
            )
            self.simulation.axes['D'].imshow(img_matches)
            self.simulation.axes['D'].tick_params(which='both',
                                                  bottom=False, left=False,
                                                  labelbottom=False, labelleft=False) 

    def after_triangulation(self, locals):
        if self.simulation.trigger_visualization:
            self.simulation.axes['E'].clear()
            self.simulation.axes['E'].set_title('Track Matching (projected onto map)')
            draw_matches_3d(ax=self.simulation.axes['E'],
                            xyz_0=locals['xyz_1'],
                            xyz_1=locals['xyz_2'],)
            self.simulation.axes['E'].set_xlim(-20, 20)
            self.simulation.axes['E'].set_ylim(0, 50)
    
    def after_track_matching(self, locals):
        img_l1, img_r1 = locals['img_pair_1']
        img_l2, img_r2 = locals['img_pair_2']
        xyz_1 = locals['xyz_1']
        xyz_2 = locals['xyz_2']
        # uv_l1, uv_r1 = self.simulation.odom.camera_model.forward(xyz_1)
        # uv_l2, uv_r2 = self.simulation.odom.camera_model.forward(xyz_2)
        uv_l1 = locals['uv_left_1']
        uv_l2 = locals['uv_left_2']
        
        # # depth map
        # depth_map = np.zeros(shape=img_l1.shape[:2])
        # for (x,y,z), (u,v) in zip(xyz_1, uv_l1):
        #     depth_map[v,u] = z
        # plt.imshow(depth_map)
        # plt.show()
        
        # optical flow
        blended_l = cv2.addWeighted(img_l1, 0.5, img_l2, 0.5, 0.0)
        
        for pt1, pt2 in zip(uv_l1, uv_l2):
            cv2.line(blended_l, pt1, pt2, color=(0,0,255), thickness=1)

        ax = self.simulation.axes['B']
        ax.clear(), ax.set_title('Optical Flow (Left Image only)')
        ax.imshow(blended_l)
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False) 
        

class Simulation():
    def __init__(self, dir, sequence, visual_odometry, T_world_to_cam_initial=np.eye(4)):
        # data
        img_stream_dir = os.path.join(dir, 'sequences', '%02d'%sequence)
        self.img_stream = KittiSlamDataloader(dir=img_stream_dir)
        
        gt_dir = os.path.join(dir, 'poses', '%02d.txt'%sequence)
        self.df_ground_truth = pd.read_csv(gt_dir, sep=' ', header=0)
        self.t_max = len(self.df_ground_truth) - 1 

        # odometry model
        self.odom = visual_odometry
        self.odom.callback = SimulationCallback(simulation=self)

        # state, to be updated
        self.t = 90
        assert self.t>0 # note: >0, because we need 2 image pairs for tracking
        Rt_gt_current = self.df_ground_truth.iloc[self.t].to_numpy().reshape(3,4)
        self._T_world_to_cam = np.eye(4)
        self._T_world_to_cam[:3, :] = Rt_gt_current
        
    
        # visualization
        self.freq = 1
        self.fig, self.axes = plt.subplot_mosaic(mosaic='AADD\nCCBB\nCCEE',
                                                 figsize=(20,10),)

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
        # state estimation
        R, t = self.T_world_to_cam[:3, :3], self.T_world_to_cam[:3, 3]
        x, z = t[0], t[2]
        self.axes['C'].scatter(x, z, c='green', s=3)
        frame_estimated, pos_estimated = draw_coordinate_frame_2d(self.axes['C'], R,t, color='green', alpha=1)

        # ground truth
        Rt_gt = self.df_ground_truth.iloc[self.t].to_numpy().reshape(3,4)
        R_gt, t_gt = Rt_gt[:,:3], Rt_gt[:,3]
        frame_gt, pos_gt = draw_coordinate_frame_2d(self.axes['C'], R_gt, t_gt, color='black', alpha=0.5)

        # camera view
        img_pair_stacked = np.concatenate(current_img_pair, axis=1)
        self.axes['A'].clear()
        self.axes['A'].imshow(img_pair_stacked, cmap='gray')
        self.axes['A'].axis('off')
        self.axes['A'].set_title('Stereo Vision (Input)')

        # pause and clean-up
        plt.tight_layout()
        plt.pause(0.01)
        pos_gt.remove(), frame_gt.remove()
        frame_estimated.remove(), pos_estimated.remove()

    def step(self):
        self.trigger_visualization = (self.t % self.freq == 0)
        
        # visual odometry
        prior_img_pair = self.img_stream[self.t-1]
        current_img_pair = self.img_stream[self.t]
        T_prior_to_current = self.odom(prior_img_pair, current_img_pair)
        # todo: tmp
        Rt_gt_current = self.df_ground_truth.iloc[self.t].to_numpy().reshape(3,4)
        Rt_gt_prior = self.df_ground_truth.iloc[self.t-1].to_numpy().reshape(3,4)
        target = homogenious_matrix(R=Rt_gt_current[:,:3], t=Rt_gt_current[:,3]) \
                 @ np.linalg.inv(homogenious_matrix(R=Rt_gt_prior[:,:3], t=Rt_gt_prior[:,3]))
        print(np.round(T_prior_to_current, decimals=2))
        print(np.round(target, decimals=2))
        # todo: tmp end
        #print('\n', np.round(T_prior_to_current, decimals=2))
        self.T_world_to_cam = self.T_world_to_cam @ T_prior_to_current
        
        if self.trigger_visualization:
            self.visualize(current_img_pair)

        self.t += 1

    def run(self):
        x_gt, z_gt = self.df_ground_truth.iloc[:, 3], self.df_ground_truth.iloc[:, 11]
        self.axes['C'].scatter(x_gt, z_gt, c='grey')
        while self.t < self.t_max:
            self.step()


if __name__ == '__main__':
    # camera model (intrinsics)
    from sensor_setup import sensor_setup
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

    from config import local_data_dir
    simulation = Simulation(dir=local_data_dir,
                            sequence=0, visual_odometry=odometry)
    
    simulation.run()




