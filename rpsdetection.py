import math
import os
import re

from mmpose.apis import MMPoseInferencer


###############
# Static      #
###############

# dictionary of labels
# key   = character used in image naming convention
# value = full label 
labels = {
    'p' : 'paper',
    'r' : 'rock',
    's' : 'scissors'
}


###############
# Functions   #
###############

# process a directory, creating a list of images and their labels
# image naming convention is {initial of contributor}{initial indicating left or right hand}{initial indicating rock/paper/scissors}{3-digit numeric}.jpg
# folder_path   = the local directory which contains the images to be processed
# output        = a list of (full image file path, ground truth label) pairs
def load_rps_images(folder_path):
    labeled_images = []
    for filename in os.listdir(folder_path):
        f = filename.lower()
        file_path = os.path.join(folder_path, f)
        if os.path.isfile(file_path):
            m = re.fullmatch('([a-z])([lr])([rps])(\d{3})\.jpg', f)
            if not m: continue #skip file if the pattern doesn't match
            lbl = m.group(3)
            if lbl in labels.keys():
                lbl = labels[lbl]
            else:
                lbl = 'unknown_' + lbl
            labeled_images.append((file_path, lbl))
    return labeled_images


###############
# Image Class #
###############

class RPSImage:
    # on construction, process a single image using MMPose, acquiring the necessary features
    def __init__(self, inferencer, img_path, out_path):
        # process the image through MMPose, saving output
        result_generator = inferencer(img_path, out_dir=out_path)
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
    
    def get_features(self):
        features = [ self.box_ratio ]
        for r in self.key_ratio:
            features.append(r)
        return features
    
    def compare_features(self, other):
        my_features = self.get_features()
        ot_features = other.get_features()
        # more to do here


###############
# Experiment  #
###############

# set local directory paths here
img_dir = 'img'
out_dir = 'out'

labeled_images = load_rps_images(img_dir)

inferencer = MMPoseInferencer('hand')
for (img_path, lbl) in labeled_images:
    img = RPSImage(inferencer, img_path, out_dir)