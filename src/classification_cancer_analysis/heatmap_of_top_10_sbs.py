"""

This file is used to draw the heatmap for display the
top 10 weighted sbs signatures in each cancer types to
form the image to conduct convolutional neural network

"""
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('..', 'my_utilities')))
from my_utilities import my_config as cfg

# given the cancer lists
from sklearn.preprocessing import MinMaxScaler

# given the cancer labels
cancer_dict = {0: 'ACC', 1: 'BLCA', 2: 'BRCA', 3: 'CESC', 4: 'CHOL', 5: 'COAD', 6: 'DLBC', 7: 'ESCA', 8: 'GBM',
               9: 'HNSC', 10: 'KICH', 11: 'KIRC', 12: 'KIRP', 13: 'LAML', 14: 'LGG', 15: 'LIHC', 16: 'LUAD', 17: 'LUSC',
               18: 'MESO', 19: 'OV', 20: 'PAAD', 21: 'PCPG', 22: 'PRAD', 23: 'READ', 24: 'SARC', 25: 'SKCM', 26: 'TGCT',
               27: 'THCA', 28: 'THYM', 29: 'UCEC', 30: 'UCS', 31: 'UVM'}

# ---------------------------------------------------------------------------------------------------------------------
# loading the cancer type sbs weight matrix from the running result(only take random fold for observation(4th fold))
cancer_type_path = './result/cancer_type-weight_4.npy'
cancer_type_weight = np.load(cancer_type_path).T  # shape (49,32)
cancer_type_scaler = MinMaxScaler()
cancer_type_nor_weight = cancer_type_scaler.fit_transform(abs(cancer_type_weight))

# normalize it to 0 and 1
cancer_type_zero_one_weight = cancer_type_nor_weight / np.sum(cancer_type_nor_weight, axis=0).reshape(1, 32)
cancer_type_zero_one_weight = cancer_type_zero_one_weight.T

# we find the top sbs in each cancer and assign 1 to then and rest as 0
cancer_top10_sbs_array = np.zeros((32, 49))
for cancer_type in range(len(cfg.ORGAN_NAMES)):
    cancer_type_zero_one_weight_c = list(cancer_type_zero_one_weight[cancer_type, :])
    # we find the top 10 weighted sbs signatures comes handy in identify this cancer
    top_10_cancer_sbs_index = list(reversed(
        sorted(range(len(cancer_type_zero_one_weight_c)), key=lambda k: cancer_type_zero_one_weight_c[k])[
        -10:]))
    for i in top_10_cancer_sbs_index:
        cancer_top10_sbs_array[cancer_type][i] = 1

# assign the sbs signature columns to the cancer type sbs weight matrix,rename the index to cancer types
cancer_df = pd.DataFrame(cancer_top10_sbs_array, columns=cfg.SBS_NAMES)
cancer_df.rename(index=cancer_dict, inplace=True)

# set up the heatmap for cancer type sbs weight matrix
plt.subplots(figsize=(20, 15))
sns.heatmap(cancer_df, annot=True, annot_kws={"size": 4}, cmap='Greys')
plt.savefig('./result/top10_sbs_cancer__heatmap/cancer_top10_sbs_heatmap.png')
