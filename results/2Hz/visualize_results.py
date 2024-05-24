import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.pardir)
from config import local_data_dir

taggings = ['Suburb', 'Highway', 'City']
fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(17, 5))

dir = '2Hz'
for i, ax in enumerate(axes):
    # ground truth
    gt_dir = os.path.join(local_data_dir, 'poses', '%02d.txt'%i)
    df_ground_truth = pd.read_csv(gt_dir, sep=' ', header=0)
    x_gt, z_gt = df_ground_truth.iloc[:, 3], df_ground_truth.iloc[:, 11]
    ax.plot(x_gt, z_gt, c='grey', label='groundtruth', lw=5)
    
    # estimation
    filename = f'logging_for_sequence_{"%02d"%i}.csv'
    filepath = os.path.join(dir, filename)
    estimation_df = pd.read_csv(filepath)
    x,z = estimation_df['x'], estimation_df['z']
    ax.plot(x,z,c='green', lw=1, label='estimation')

    ax.set_title(f'Sequence {"%02d"%i} ({taggings[i]})')
    if i == 0:
        ax.legend(fancybox=False, edgecolor='black')
plt.show()