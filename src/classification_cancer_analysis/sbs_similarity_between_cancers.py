""""

This file is used to investigate on why has the sbs signature has caused some of the samples of a class classified to
other classes,the reason is that some of the classes are sharing the similar sbs signature exposures

"""
import os
import sys
import numpy as np
from scipy import spatial
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('..', 'my_utilities')))
from my_utilities import my_config as cfg

cancer_list = cfg.ORGAN_NAMES
# we only do it for 4th fold (only take random fold for observation(4th fold))
cancer_type_path = './result/cancer_type-weight_4.npy'
cancer_type_weight = np.load(cancer_type_path).T  # shape (49,32)
cancer_type_scaler = MinMaxScaler()
cancer_type_nor_weight = cancer_type_scaler.fit_transform(abs(cancer_type_weight))

# normalize it to 0 and 1
cancer_type_zero_one_weight = cancer_type_nor_weight / np.sum(cancer_type_nor_weight, axis=0).reshape(1, 32)

cancer_similarities = {}

# calculate the cosine similarity for each pairwise cancers
for i in range(32):
    other_cancers = []
    for j in range(32):
        other_cancers.append(
            1 - spatial.distance.cosine(cancer_type_zero_one_weight[i], cancer_type_zero_one_weight[j]))
        cancer_similarities[i] = other_cancers


cancer_similarities_list = []
for cancer in cancer_similarities:
    cancer_similarities_list.append(cancer_similarities[cancer])

num = 0
# find the top 5 similar cancers and their cosine similarity with each cancers and print them for experimental
# support
for cancer_lst in cancer_similarities_list:
    index = list(reversed(sorted(range(len(cancer_lst)), key=lambda i: cancer_lst[i])[-5:]))
    print(cancer_list[num], [cancer_list[x] for x in index], [cancer_lst[y] for y in index])
    num += 1

# convert the list to dataframe and plot the similarity with seaborn
cancer_similarity_df = pd.DataFrame(cancer_similarities_list, columns=cancer_list)
cancer_similarity_df.rename(index=cfg.cancer_dict, inplace=True)

plt.subplots(figsize=(20, 15))
sns.heatmap(cancer_similarity_df, annot=True, annot_kws={"size": 4}, cmap='Reds')
plt.savefig('./result/cancer_cancer_similarity_heatmaps/cancer_cancer_heatmap.png')
