""""

This file is used to investigate on why has the sbs signature has caused some of the samples of a class classified to
other classes,the reason is that some of the classes are sharing the similar sbs signature exposures

"""
import numpy as np
from scipy import spatial
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

cancer_list = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP',
               'LAML',
               'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'TGCT',
               'THCA',
               'THYM', 'UCEC', 'UCS', 'UVM']

# given the cancer labels
cancer_dict = {0: 'ACC', 1: 'BLCA', 2: 'BRCA', 3: 'CESC', 4: 'CHOL', 5: 'COAD', 6: 'DLBC', 7: 'ESCA', 8: 'GBM',
               9: 'HNSC', 10: 'KICH', 11: 'KIRC', 12: 'KIRP', 13: 'LAML', 14: 'LGG', 15: 'LIHC', 16: 'LUAD', 17: 'LUSC',
               18: 'MESO', 19: 'OV', 20: 'PAAD', 21: 'PCPG', 22: 'PRAD', 23: 'READ', 24: 'SARC', 25: 'SKCM', 26: 'TGCT',
               27: 'THCA', 28: 'THYM', 29: 'UCEC', 30: 'UCS', 31: 'UVM'}

cancer_sbs_weight = np.load("./result/cancer_type_normalized-weight.npy", mmap_mode='r').T

cancer_similarities = {}

for i in range(32):
    other_cancers = []
    for j in range(32):
        other_cancers.append(1 - spatial.distance.cosine(cancer_sbs_weight[i], cancer_sbs_weight[j]))
        cancer_similarities[i] = other_cancers

# print(cancer_similarities)

cancer_similarities_list = []
for cancer in cancer_similarities:
    cancer_similarities_list.append(cancer_similarities[cancer])

num = 0
for cancer_lst in cancer_similarities_list:
    index = list(reversed(sorted(range(len(cancer_lst)), key=lambda i: cancer_lst[i])[-5:]))
    print(cancer_list[num], [cancer_list[x] for x in index], [cancer_lst[y] for y in index])
    num += 1

cancer_similarity_df = pd.DataFrame(cancer_similarities_list, columns=cancer_list)
cancer_similarity_df.rename(index=cancer_dict, inplace=True)

plt.subplots(figsize=(20, 15))
sns.heatmap(cancer_similarity_df, annot=True, annot_kws={"size": 4}, cmap='Reds')
plt.savefig('./result/cancer_cancer_similarity_heatmaps/cancer_cancer_heatmap.png')
