"""

This file is used to draw the heatmap for display the
weight of sbs signatures in each cancer types

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# loading the cancer type sbs weight matrix from the running result
cancer_sbs_weight = np.load("result/cancer_type_normalized-weight.npy", mmap_mode='r').T

# given the sbs columns
SBS_NAMES_lst = ['SBS4', 'SBS5', 'SBS1', 'SBS39', 'SBS36', 'SBS2', 'SBS13', 'SBS10b', 'SBS9', 'SBSPON', 'SBS3', 'SBS6',
                 'SBS30',
                 'SBSN', 'SBS10a', 'SBS15', 'SBS26', 'SBS29', 'SBS17b', 'SBS87', 'SBS16', 'SBS18', 'SBS52', 'SBS8',
                 'SBS7b', 'SBS40',
                 'SBS50', 'SBS24', 'SBS27', 'SBS42', 'SBS86', 'SBS57', 'SBS33', 'SBS90', 'SBS17a', 'SBS55', 'SBS22',
                 'SBS54', 'SBS48',
                 'SBS58', 'SBS28', 'SBS7a', 'SBS7d', 'SBS7c', 'SBS38', 'SBS84', 'SBS35', 'SBS14', 'SBS44']

# given the cancer labels
cancer_dict = {0: 'BLCA', 1: 'LGG'}


# assign the sbs signature columns to the cancer type sbs weight matrix,rename the index to cancer types
cancer_df = pd.DataFrame(cancer_sbs_weight, columns=SBS_NAMES_lst)
cancer_df.rename(index=cancer_dict, inplace=True)
# set up the heatmap for cancer type sbs weight matrix
plt.subplots(figsize=(20, 15))
sns.heatmap(cancer_df, annot=True, annot_kws={"size": 4}, cmap="RdYlGn")
plt.savefig('result/sbs-weight-heatmap/cancer_sbs_heatmap.png')
