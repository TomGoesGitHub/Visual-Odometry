import os
import cv2

class KittiSlamDataloader:
    '''A Dataloader suited for the visual-odometry problem on the KittiSlam Dataset.
    Loads preprocessed images from source-directory.
    See: https://stackoverflow.com/questions/42983569/how-to-write-a-generator-class
    '''
    def __init__(self, dir):
        '''
        Args:
            dir: directory containing the image-data. (e.g. path-to-data\00)
        '''        
        self.dir = dir
    
    def __getitem__(self, key):
        '''
        Args:
            key: timestep-index

        Returns:
            image tuple: left and right image of current timestep
        '''
        path_left = os.path.join(self.dir, 'image_0', '%06d.png'%key)
        path_right = os.path.join(self.dir, 'image_1', '%06d.png'%key)
        img_left, img_right = cv2.imread(path_left), cv2.imread(path_right)
        img_left, img_right = self.preprocess(img_left), self.preprocess(img_right)
        return img_left, img_right
    
    def preprocess(self, img):
        '''Preprocessing. Use one channel only and apply histogram-equalization.

        Args:
            img: raw input-image

        Returns:
            preprocessed image
        '''
        img, _, _ = cv2.split(img)
        img = cv2.equalizeHist(img)
        return img