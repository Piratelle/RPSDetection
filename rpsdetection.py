#region Imports
import math
import numpy as np
import os
import re

from mmpose.apis import MMPoseInferencer
#endregion


#region Image Class
###############
# Image Class #
###############

class RPSImage:
    HANDS = { 'l' : 'left', 'r' : 'right' }
    LABELS = { 'p' : 'paper', 'r' : 'rock', 's' : 'scissors' }
    
    # on construction, process a single image using MMPose, acquiring the necessary features
    def __init__(self, inferencer, file_name, file_path, out_path):
        # save path and label
        self.name = file_name.lower()
        self.path = file_path
        self.label, self.hand = '', ''
        self.learn_labels()
        
        # process the image through MMPose, saving output
        result_generator = inferencer(self.path, out_dir=out_path)
        result = next(result_generator)
        data_dict = result['predictions'][0][0]
        
        # translate results into features
        ## bounding box ratio: y-dim / x-dim of bounding box
        bbox = data_dict['bbox'][0]
        self.box_ratio = (bbox[3] - bbox[1]) / (bbox[2] - bbox[0])
        
        ## keypoints ratios[0-5]: y-dim / x-dim of keypoint range, for thumb - pinky and overall
        kp_range = [[1,4],[5,8],[9,12],[13,16],[17,20],[0,20]]
        range_count = len(kp_range)
        min_x = [math.inf for i in range(range_count)]
        min_y = [math.inf for i in range(range_count)]
        max_x = [0 for i in range(range_count)]
        max_y = [0 for i in range(range_count)]
        self.key_ratio = [0 for i in range(range_count)]
        keypoints = data_dict['keypoints']
        for j in range(range_count):
            min_val, max_val = kp_range[j]
            for i in range(len(keypoints)):
                if min_val <= i <= max_val:
                    x, y = keypoints[i]
                    if x < min_x[j]: min_x[j] = x
                    if x > max_x[j]: max_x[j] = x
                    if y < min_y[j]: min_y[j] = y
                    if y > max_y[j]: max_y[j] = y
            if (max_x[j] != min_x[j]): # prevent divide-by-zero errors
                self.key_ratio[j] = (max_y[j] - min_y[j]) / (max_x[j] - min_x[j])
    
    # learn the ground truth labels from this image's file name
    # naming convention is {initial of contributor}{initial indicating left or right hand}{initial indicating rock/paper/scissors}{3?-digit numeric}.jpg
    def learn_labels(self):
        m = re.fullmatch('([a-z])([lr])([rps])(\d+)\.jpg', self.name)
        if not m: return
        
        lbl = m.group(3)
        if lbl in self.LABELS.keys():
            self.label = self.LABELS[lbl]
        else:
            self.label = 'unknown_' + lbl
        
        hand = m.group(2)
        if hand in self.HANDS.keys():
            self.hand = self.HANDS[hand]
        else:
            self.hand = 'unknown'
    
    # return an array of labels in order of categorical relevance
    def get_labels(self):
        return [ self.label, self.hand ]
    
    # return an array of feature values
    def get_features(self):
        features = [ self.box_ratio ]
        for r in self.key_ratio:
            features.append(r)
        return features
    
    # return the result of comparing this image's features with another RPSImage's features
    def compare_features(self, other):
        my_features = self.get_features()
        ot_features = other.get_features()
        l2 = np.linalg.norm(my_features - ot_features) # calculate L2 distance
        return l2

#endregion


#region Functions
###############
# Functions   #
###############

# process a directory of images, learning their ground truths and features
# folder_path   = the local directory which contains the images to be processed
# output        = a list of featured image objects
def load_rps_images(folder_path, out_path):
    featured_images = []
    inferencer = MMPoseInferencer('hand')
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            featured_images.append(RPSImage(inferencer, file_name, file_path, out_path))
    return featured_images

def build_train_test(train_pct):
    return
#endregion


#region Experiment
###############
# Experiment  #
###############

# set local directory paths here
img_dir = 'img'
out_dir = 'out'

# build featured image objects from images in directory
images = load_rps_images(img_dir, out_dir)

#endregion