import os
import re

from mmpose.apis import MMPoseInferencer

# dictionary of labels
# key   = character used in image naming convention
# value = full label 
labels = {
    'p' : 'paper',
    'r' : 'rock',
    's' : 'scissors'
}

# process a directory, creating a list of images and their labels
# image naming convention is {initial of contributor}{initial indicating left or right hand}{initial indicating rock/paper/scissors}{3-digit numeric}.jpg
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
            


#img_path = 'img/test.jpg'   # replace this with your own image path

# instantiate the inferencer using the model alias
#inferencer = MMPoseInferencer('hand')

# The MMPoseInferencer API employs a lazy inference approach,
# creating a prediction generator when given input
#result_generator = inferencer(img_path, show=True, out_dir='out')
#result = next(result_generator)

## run code here!
labeled_images = load_rps_images('img')
print(labeled_images)