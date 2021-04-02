"""

This file is used to draw the heatmap for display the
weight of sbs signatures in each cancer types

"""
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('..', 'my_utilities')))
from my_utilities import my_config as cfg

# load the file first
# The loading of cancer's sbs weights start from here (only take random fold for observation(4th fold))
cancer_type_path = './result/cancer_type-weight_4.npy'
cancer_type_weight = np.load(cancer_type_path).T  # shape (49,32)
cancer_type_scaler = MinMaxScaler()
# we don't take absolute value here cause we only want the positive contribution of sbs to that cancer
cancer_type_nor_weight = cancer_type_scaler.fit_transform(cancer_type_weight)

# The normalization of weights of sbs signatures in cancer types for heatmap generating
# normalize it to 0 and 1
cancer_type_zero_one_weight = cancer_type_nor_weight / np.sum(cancer_type_nor_weight, axis=0).reshape(1, 32)

# assign the sbs signature columns to the cancer type sbs weight matrix,rename the index to cancer types
cancer_df = pd.DataFrame(cancer_type_zero_one_weight.T, columns=cfg.SBS_NAMES)
cancer_df.rename(index=cfg.cancer_dict, inplace=True)
print(cancer_df)

# set up the heatmap for cancer type sbs weight matrix
plt.subplots(figsize=(20, 15))
sns.heatmap(cancer_df, annot=True, annot_kws={"size": 4}, cmap='Reds')
plt.savefig('./result/weight_heatmaps/cancer_sbs_heatmap.png')