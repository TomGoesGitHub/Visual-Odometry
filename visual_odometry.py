import cv2
import numpy as np

from utils.features import filter_horizontal_matches, filter_matches_by_distance_ratio, filter_best_matches
from utils.spatial import homogenious_matrix


class StereoCamera:
    '''StereoCamera class. Stereo vision is characterized by its two cameras in use.
    This class asumes that both cameras have identical properties.'''
    def __init__(self, baseline, focal_len, center_pos):
        '''
        Args:
            baseline: Distance between left and right camera in meter
            focal_len: focal length in pixels
            center_pos: position of the image center in pixels (measured relative
            to the left top corner)
        '''        
        # intrinsics
        self.baseline = baseline # in meter
        self.focal_len = focal_len # in pixels
        self.center_pos = center_pos # in pixels, in pixel coordinates

    def intrincsic_matrix(self, which='left'):
        '''Returns the intrinsic camera matrix of the specified camera.

        Args:
            which (str, optional): Wether to query the left or right camera. Defaults to 'left'.

        Returns:
            np.array: The 3x4 camera matrix.
        '''        
        assert which in ['left', 'right']

        c_u, c_v, f_u, f_v = *self.center_pos, *self.focal_len 
        b = self.baseline if (which == 'right') else 0

        return np.array([[f_u, 0, c_u, -f_u*b],
                         [0, f_v, c_v,      0],
                         [0,   0,   1,      0]])
        
    
    def triangulate(self, uv_left, uv_right):
        ''' Inverse camera model: Recover the depth from the stereo vision pixels.    
        See: First Principles of Computer Vision - Simple Stereo | Camera Calibration
        https://www.youtube.com/watch?v=hUVyDabn1Mg&t=330s

        Args:
            uv_left: pixel coordinates of the keypoints in the left image
            uv_right: pixel coordinates of the keypoints in the right image

        Returns:
            spatial xyz-position of the keypoints relative to the camera coordinate system
        '''              
        u_l, v_l = uv_left.T
        u_r, v_r = uv_right.T
        
        b, c_u, c_v, f_u, f_v = self.baseline, *self.center_pos, *self.focal_len # alias

        d = u_l - u_r # disparity
        d = np.clip(d, a_min=1e-6, a_max=np.inf) # to avoid dividing by zero
        x = b * (u_l - c_u) / d
        y = b * f_u/f_v * (v_l - c_v) / d
        z = b * f_u / d
        
        xyz = np.stack([x,y,z], axis=1)
        return xyz
    
    def forward(self, xyz):
        '''Forward camera model: Get pixel coordinates from 3D point.
        See: Geiger, 2013: Vision Meets Robotics - The Kitti Dataset, Eq. 4.

        Args:
            xyz: spatial 3D-position of the keypoint relative to the camera coordinate system

        Returns:
            pixel coordinate tuple: pixel coordinates of the keypoints projection
            onto the left/right image
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
    '''Abstract Callback Class for Feature Based Visual Odometry. By subclassing, additional
    functionality can be added at the corresponding points during runtime.
    '''
    def after_stereo_matching(self, locals):
        pass
    
    def after_triangulation(self, locals):
        pass

    def after_track_matching(self, locals):
        pass   


class FeatureBasedVisualOdemetry:
    '''Performs Visual Odemetry (Relative Pose Estimation from image input only) by identifying
    and tracking keypoints in the images which are described by a set of features.'''
    def __init__(self, detector, descriptor, matcher, camera_model,
                 callback=FeatureBasedVisualOdometryCallback()):
        self.detector = detector
        self.descriptor = descriptor
        self.matcher = matcher
        self.camera_model = camera_model
        self.callback = callback
    
    def match_stereo(self, img_l, img_r, kp_l, kp_r):
        ''' Pipeline for the computation and flitering of stereo-matches between 
        left and right image. The goal is to keep good matches only.
        See: Laganiere - Computer vision application programming cookbook, Chapter 9.
        '''        
        _, des_l = self.descriptor.compute(img_l, kp_l)
        _, des_r = self.descriptor.compute(img_r, kp_r)
        
        knn_matches = self.matcher.knnMatch(des_l, des_r, k=2)
        matches = filter_matches_by_distance_ratio(knn_matches)
        matches = filter_horizontal_matches(kp_l, kp_r, matches)
        #matches = filter_best_matches(matches, keep_n_best=100)
        self.callback.after_stereo_matching(locals())

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
        and 3D reconstruction.'''
        # keypoint detection
        kp_left, kp_right = self.detector.detect(img_left), self.detector.detect(img_right)

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
        
        return xyz_cam, des, uv_left
    
    def __call__(self, img_pair_1, img_pair_2):
        '''Given an image pair for each camera position, calculate the estimation (best
        fit) of the spatial transformation between the 2 camera positions.

        Args:
            img_pair_1: image tuple (left and right image) for prior timestep t1
            img_pair_2: image tuple (left and right image) for sucessor timestep t2

        Returns:
            np.array: 4x4 transformation matrix describing the relative movement of
            the camera during the period from t1 to t2.
        '''
        xyz_1, des1, uv_left_1 = self.run_stereo_vision_pipeline(*img_pair_1)
        xyz_2, des2, uv_left_2 = self.run_stereo_vision_pipeline(*img_pair_2)
        
        tracking_matches = self.match_track(des1, des2)
        idx_1 = np.array([match.queryIdx for match in tracking_matches])
        idx_2 = np.array([match.trainIdx for match in tracking_matches])
        xyz_1, xyz_2 = xyz_1[idx_1], xyz_2[idx_2]
        uv_left_1, uv_left_2 = uv_left_1[idx_1], uv_left_2[idx_2]
        self.callback.after_track_matching(locals())

        intrinsics = self.camera_model.intrincsic_matrix(which='left')
        
        try:
            _, rvec, tvec, is_inlier = cv2.solvePnPRansac(objectPoints=xyz_2,
                                                        imagePoints=uv_left_1.astype(float),
                                                        # note: cv2 requires float here, which is weird...
                                                        cameraMatrix=intrinsics[:, :3],
                                                        distCoeffs=None)
            rmat = cv2.Rodrigues(rvec)[0]    
            T_1_2 = homogenious_matrix(R=rmat, t=tvec)
            is_inlier = is_inlier.squeeze()

            xyz_1, xyz_2 = xyz_1[is_inlier], xyz_2[is_inlier]
            self.callback.after_triangulation(locals())
        except cv2.error as e:
            print('solving Ransac was not sucessful. Skipping current frame.')
            T_1_2 = homogenious_matrix() # identy-matrix

        return T_1_2

    def tile(self, img, nrows, ncols):
        '''Divide the image into smaller tiles (tiles can be thought of as smaller images themself)'''
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
        return tiles, uv

    def get_keypoints_from_tiles(self, img, nrows=10, ncols=30):
        '''Tile the image and compute keypoints for each tile individually. Compared to the
        keypoint computation without tiling this can help to obtain an uniform keypoint distribution over
        the original image.'''
        tiles, uv = self.tile(img, nrows, ncols)
        kp_img = []
        
        for tile, (u,v) in zip(tiles, uv):
            # compute keypoints in tile
            kp_tile = self.detector.detect(tile)

            # correct coordinates (from tile pxiel coorinates to img pixel coordinates)
            for kp in kp_tile:
                kp.pt = (kp.pt[0]+u, kp.pt[1]+v)
            kp_img.extend(kp_tile)
        
        return kp_img        

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
        