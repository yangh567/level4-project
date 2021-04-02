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
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('..', 'my_utilities')))
from my_utilities import my_config as cfg

# given the cancer labels
cancer_dict = {0: 'ACC', 1: 'BLCA', 2: 'BRCA', 3: 'CESC', 4: 'CHOL', 5: 'COAD', 6: 'DLBC', 7: 'ESCA', 8: 'GBM',
               9: 'HNSC', 10: 'KICH', 11: 'KIRC', 12: 'KIRP', 13: 'LAML', 14: 'LGG', 15: 'LIHC', 16: 'LUAD', 17: 'LUSC',
               18: 'MESO', 19: 'OV', 20: 'PAAD', 21: 'PCPG', 22: 'PRAD', 23: 'READ', 24: 'SARC', 25: 'SKCM', 26: 'TGCT',
               27: 'THCA', 28: 'THYM', 29: 'UCEC', 30: 'UCS', 31: 'UVM'}

# ---------------------------------------------------------------------------------------------------------------------
# loading the cancer type sbs weight matrix from the running result
cancer_sbs_weight = np.load("./result/cancer_type_normalized-weight.npy", mmap_mode='r').T

# assign the sbs signature columns to the cancer type sbs weight matrix,rename the index to cancer types
cancer_df = pd.DataFrame(cancer_sbs_weight, columns=cfg.SBS_NAMES)
cancer_df.rename(index=cancer_dict, inplace=True)
print(cancer_df)

# set up the heatmap for cancer type sbs weight matrix
plt.subplots(figsize=(20, 15))
sns.heatmap(cancer_df, annot=True, annot_kws={"size": 4}, cmap='Reds')
plt.savefig('./result/weight_heatmaps/cancer_sbs_heatmap.png')