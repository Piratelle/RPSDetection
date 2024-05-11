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
        
        # translate results into features


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