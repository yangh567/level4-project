"""

This file is used to draw the heatmap for display the
weight of sbs signatures in each cancer types or gene types

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# loading the cancer type sbs weight matrix from the running result
cancer_sbs_weight = np.load("../result/cancer_type_normalized-weight.npy", mmap_mode='r').T

# loading the gene sbs weight matrix from the running result
gene_sbs_weight = np.load("../result/gene_normalized-weight.npy", mmap_mode='r').T

# given the sbs columns
SBS_NAMES_lst = ['SBS4', 'SBS5', 'SBS1', 'SBS39', 'SBS36', 'SBS2', 'SBS13', 'SBS10b', 'SBS9', 'SBSPON', 'SBS3', 'SBS6',
                 'SBS30',
                 'SBSN', 'SBS10a', 'SBS15', 'SBS26', 'SBS29', 'SBS17b', 'SBS87', 'SBS16', 'SBS18', 'SBS52', 'SBS8',
                 'SBS7b', 'SBS40',
                 'SBS50', 'SBS24', 'SBS27', 'SBS42', 'SBS86', 'SBS57', 'SBS33', 'SBS90', 'SBS17a', 'SBS55', 'SBS22',
                 'SBS54', 'SBS48',
                 'SBS58', 'SBS28', 'SBS7a', 'SBS7d', 'SBS7c', 'SBS38', 'SBS84', 'SBS35', 'SBS14', 'SBS44']

# given the cancer labels
cancer_dict = {0: 'ACC', 1: 'BLCA', 2: 'BRCA', 3: 'CESC', 4: 'CHOL', 5: 'COAD', 6: 'DLBC', 7: 'ESCA', 8: 'GBM',
               9: 'HNSC', 10: 'KICH', 11: 'KIRC', 12: 'KIRP', 13: 'LAML', 14: 'LGG', 15: 'LIHC', 16: 'LUAD', 17: 'LUSC',
               18: 'MESO', 19: 'OV', 20: 'PAAD', 21: 'PCPG', 22: 'PRAD', 23: 'READ', 24: 'SARC', 25: 'SKCM', 26: 'TGCT',
               27: 'THCA', 28: 'THYM', 29: 'UCEC', 30: 'UCS', 31: 'UVM'}

# given the gene labels
gene_dict = {0: 'CCND1', 1: 'CCND2', 2: 'CCND3', 3: 'CCNE1', 4: 'CDK4', 5: 'CDK6', 6: 'E2F1', 7: 'E2F3', 8: 'YAP1',
             9: 'MYC', 10: 'MYCN', 11: 'ARRDC1', 12: 'KDM5A', 13: 'NFE2L2', 14: 'AKT1', 15: 'AKT2', 16: 'PIK3CA',
             17: 'PIK3CB', 18: 'PIK3R2', 19: 'RHEB', 20: 'RICTOR', 21: 'RPTOR', 22: 'EGFR', 23: 'ERBB2', 24: 'ERBB3',
             25: 'PDGFRA', 26: 'MET', 27: 'FGFR1', 28: 'FGFR2', 29: 'FGFR3', 30: 'FGFR4', 31: 'KIT', 32: 'IGF1R',}

# assign the sbs signature columns to the cancer type sbs weight matrix,rename the index to cancer types
cancer_df = pd.DataFrame(cancer_sbs_weight, columns=SBS_NAMES_lst)
cancer_df.rename(index=cancer_dict, inplace=True)

# assign the sbs signature columns to the gene sbs weight matrix,rename the index to genes
gene_df = pd.DataFrame(gene_sbs_weight, columns=SBS_NAMES_lst)
gene_df.rename(index=gene_dict, inplace=True)

print(cancer_df)
print(gene_df)

# set up the heatmap for cancer type sbs weight matrix
plt.subplots(figsize=(20, 15))
sns.heatmap(cancer_df, annot=True, annot_kws={"size": 4}, cmap="Reds")
plt.savefig('./figures/cancer_sbs_heatmap.png')

# set up the heatmap for gene sbs weight matrix
plt.subplots(figsize=(20, 15))
sns.heatmap(gene_df, annot=True, annot_kws={"size": 4}, cmap="Reds")
plt.savefig('./figures/gene_sbs_heatmap.png')
