#region Imports
import heapq
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random as rnd
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
        l2 = np.linalg.norm(np.array(my_features) - np.array(ot_features)) # calculate L2 distance
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

# given a dataset and a training percentage, randomly split the data into training and testing sets
def build_train_test(train_pct, dataset):
    # traditionally the bulk of the dataset is used for training, 
    # so start with all train then build up test
    test = []
    train = dataset.copy()
    while len(train) / len(dataset) > train_pct:
        item = rnd.choice(train)
        test.append(item)
        train.remove(item)
    return train, test

# given a training set, a test image, and k, acquire the k neareest neighbor set
def get_knn(train, img, k):
    neighbors = []
    for t_img in train:
        heapq.heappush(neighbors, (img.compare_features(t_img), t_img))
    return [heapq.heappop(neighbors) for i in range(k)]
    
# given a test image, and its k nearest neighbors, predict its label and check for accuracy
# output    = a list of (ground truth label, knn predicted label, knn vote percentage, success boolean) tuples
def predict(img, k_neighbors):
    # check accuracy for each level of label categorization (shape, shape+hand, etc.)
    true_lbls = img.get_labels()
    results = []
    k = len(k_neighbors)
    for i in range(len(true_lbls)):
        ground = " ".join(true_lbls[0:i+1])
        votes = {}
        for dist, n in k_neighbors:
            vote_lbl = " ".join(n.get_labels()[0:i+1])
            votes[vote_lbl] = votes.get(vote_lbl, 0) - 1 # so we can use a min-heap, the default of heapq
        tally = [(votes[vote], vote) for vote in votes]
        #print(f"\t\ttally={tally}")
        heapq.heapify(tally)
        win_tot, win_val = heapq.heappop(tally)
        #print(f"\t\twinner={win_val}")
        results.append((ground, win_val, (-1 * win_tot) / k, ground == win_val))
    return results

# clamp value to be within a minimum and maximum boundary (inclusive)
def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

# clamp value to be a percentage (0  to 100)
def clamp_pct(val):
    return clamp(val, 0, 100)

# perform the experiment!
# this is a function to allow early return in error conditions
def experiment(img_dir, out_dir, max_k, train_min, train_max, train_inc):
    # build featured image objects from images in directory
    # the objects calculate their features once to save computational time during the train/test phase
    print("* * * BUILD PHASE * * *")
    images = load_rps_images(img_dir, out_dir)
    if len(images) == 0:
        print("Error: no valid images detected in " + img_dir)
        return
    
    # clamp parameters before using, just to be safe
    tmin = clamp_pct(train_min)
    tmax = clamp_pct(train_max)
    tinc = min(train_inc, tmax - tmin)
    
    # prepare a header for the results
    lbl_count = len(images[0].get_labels())
    hdr = [ "Image", "Train Pct", "K" ]
    for i in range(lbl_count):
        depth_tag = f" w Depth {(i+1)}"
        add_hdr = [lbl + depth_tag for lbl in ["Ground Truth","Prediction","KNN Confidence","Success"]]
        hdr += add_hdr
    
    # test with a range of different training percentages
    print("* * * TRAIN/TEST PHASE * * *")
    data = []
    for train_pct in range(tmin, tmax + 1, tinc):
        print(f"starting test w/ train pct = {train_pct}")
        train, test = build_train_test(train_pct / 100, images)
        test_k = clamp(max_k, 1, len(train)) # just in case
        
        for test_img in test:
            # get the maximum, then slice to avoid repeated heapification
            knn = get_knn(train, test_img, test_k)
            
            # test with a range of different k values
            for k in range(test_k):
                #print(f"\tknn w/ k = {k + 1}")
                prediction = predict(test_img, knn[0:k + 1])
                data_row = [ test_img.name, train_pct, (k + 1) ]
                for result in prediction:
                    data_row = [*data_row, *result]
                data.append(data_row)
    
    #return results in a DataFrame
    return pd.DataFrame(data, columns=hdr)
           
#endregion


#region Experiment
###############
# Experiment  #
###############

# set local directory paths here
src_dir = 'img'
mmpose_out_dir = 'out'

# set experimental parameters here
max_k = 7 # min is 1
min_train, max_train = 75, 90 # percentages
train_inc = 5

# all the work gets done here, results in a DataFrame which is then saved to CSV
results = experiment(src_dir, mmpose_out_dir, max_k, min_train, max_train, train_inc)
results.to_csv('results.csv', index=False)

#endregion


#region Visualizations
###############
# Visuals     #
###############

# plot the results for different training percentages and k values, splitting out by categorical depth
agg = results[['Train Pct','K','Success w Depth 1','Success w Depth 2']].groupby(['Train Pct', 'K']).agg({'count', lambda gr: np.count_nonzero(gr)}).reset_index()
agg['Depth 1 Pct'] = agg['Success w Depth 1','<lambda_0>'] / agg['Success w Depth 1','count']
agg['Depth 2 Pct'] = agg['Success w Depth 2','<lambda_0>'] / agg['Success w Depth 2','count']

pcts = sorted(results['Train Pct'].unique())
depth1 = agg[agg['Train Pct',''] == pcts[0]][['K']]
depth2 = depth1.copy()

fig = plt.figure(figsize=(10, 5))
gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
(ax1, ax2) = gs.subplots(sharex=True, sharey=True)
fig.suptitle('Success Rate Trends')
ax1.set_title('Shape Only')
ax2.set_title('Shape & Hand')
ax1.set(xlabel='k', ylabel='% Correct')
ax2.set(xlabel='k')

for pct in pcts:
  filtered = agg[agg['Train Pct',''] == pct].reset_index()
  lbl = f'Training Pct {pct}'
  depth1[lbl] = filtered['Depth 1 Pct','']
  depth2[lbl] = filtered['Depth 2 Pct','']

  ax1.plot(depth1['K'], depth1[lbl], label=lbl)
  ax2.plot(depth2['K'], depth2[lbl], label=lbl)

plt.legend()
plt.savefig('success.png')
plt.show()


# plot the results for different training percentages and k values at the first categorical depth, splitting out by shape
agg_shape = results[['Train Pct','K','Ground Truth w Depth 1','Success w Depth 1','Success w Depth 2']].groupby(['Train Pct', 'K','Ground Truth w Depth 1']).agg({'count', lambda gr: np.count_nonzero(gr)}).reset_index()
agg_shape['Depth 1 Pct'] = agg_shape['Success w Depth 1','<lambda_0>'] / agg_shape['Success w Depth 1','count']

lbls = sorted(results['Ground Truth w Depth 1'].unique())
depths = [ agg_shape[(agg_shape['Train Pct',''] == pcts[0]) & (agg_shape['Ground Truth w Depth 1',''] == lbls[0])][['K']].reset_index() ]
for i in range(1, len(lbls)):
  depths.append(depths[0].copy())

fig = plt.figure(figsize=(12, 5))
gs = fig.add_gridspec(1, 3, hspace=0, wspace=0)
(ax1, ax2, ax3) = gs.subplots(sharex=True, sharey=True)
fig.suptitle('Success Rate Trends By Shape')
axes = [ax1, ax2, ax3]
ax1.set(xlabel='k', ylabel='% Correct')
ax2.set(xlabel='k')
ax3.set(xlabel='k')
for i in range(len(lbls)):
  lbl = lbls[i]
  ax = axes[i]
  ax.set_title(lbl)
  depth = depths[i]
  for pct in pcts:
    filtered = agg_shape[(agg_shape['Train Pct',''] == pct) & (agg_shape['Ground Truth w Depth 1',''] == lbl)].reset_index()
    col_lbl = f'Training Pct {pct}'
    depth[col_lbl] = filtered['Depth 1 Pct','']
    
    ax.plot(depth['K'], depth[col_lbl], label=col_lbl)

plt.legend()
plt.savefig('success_shapes.png')
plt.show()

#endregion