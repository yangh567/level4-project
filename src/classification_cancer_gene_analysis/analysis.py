"""
The Analysis of cancer types and gene types
"""
import sys,os
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('..','my_utilities')))
#import my_utilities as my_u
from my_utilities import my_config as cfg
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# load the file first

# The loading of cancer's sbs weights start from here

cancer_type_path = './result/cancer_type-weight_0.npy'
cancer_type_weight = np.load(cancer_type_path).T  # shape (49,32)
cancer_type_scaler = MinMaxScaler()
cancer_type_nor_weight = cancer_type_scaler.fit_transform(cancer_type_weight)
# normalize it to 0 and 1
cancer_type_zero_one_weight = cancer_type_nor_weight / np.sum(cancer_type_nor_weight, axis=0).reshape(1, 32)
# save the data
np.save("./result/cancer_type_normalized-weight.npy", cancer_type_zero_one_weight)

# this will be used to find sbs weights
cancer_type_zero_one_weight = cancer_type_zero_one_weight.T
# calculate the mean of the weight and find the sbs signatures have the weight greater than mean
cancer_type_mean = np.std(cancer_type_zero_one_weight, axis=1)
for i, item in enumerate(cancer_type_zero_one_weight):
    item[item > cancer_type_mean[i]] = 1
    item[item <= cancer_type_mean[i]] = 0

# extract out the sbs signatures that have the weight as 1 in cancer types
cancer_type_sbs = []

for i, item in enumerate(cancer_type_zero_one_weight):
    sub_sbs = []
    for j, sub_item in enumerate(item):
        if sub_item == 1.:
            sub_sbs.append(cfg.SBS_NAMES[j])
    cancer_type_sbs.append(sub_sbs)

# The mutated frequency is required again to find top 5 gene names for specific cancer
gene_prob = pd.read_csv('../statistics/gene_distribution/gene_prob.csv')
cancer_prob = {}
for name, item in gene_prob.groupby('cancer type'):
    cancer_prob[name] = item


# The loading of gene's sbs weights start from here
cancer_list = cfg.ORGAN_NAMES
# set up the number for each cancer_type
k = 0

result = [['cancer type', 'genes']]

for cancer_type in cancer_list:
    # used for constructing and saving result data frame later
    gene_path = './result/gene_sbs_weights/gene_type-weight_in_fold0_for_' + cancer_type + '.npy'

    gene_weight = np.load(gene_path).T  # shape (49, 10)

    # standardize the weight
    gene_scaler = MinMaxScaler()

    gene_nor_weight = gene_scaler.fit_transform(gene_weight)

    # normalize it to 0 and 1
    gene_zero_one_weight = gene_nor_weight / np.sum(gene_nor_weight, axis=0).reshape(1, 5)
    np.save(
        "./result/gene_sbs_weights/gene_normalized_weights_for_each_cancer/gene_normalized-weight_%s.npy" % cancer_type,
        gene_zero_one_weight)

    gene_zero_one_weight = gene_zero_one_weight.T
    # calculate the mean of the weight and find the sbs signatures have the weight greater than mean
    gene_mean = np.std(gene_zero_one_weight, axis=1)
    for i, item in enumerate(gene_zero_one_weight):
        item[item > gene_mean[i]] = 1
        item[item <= gene_mean[i]] = 0

    # extract out the sbs signatures that have the weight as 1 in gene types
    gene_sbs = []
    for i, item in enumerate(gene_zero_one_weight):
        sub_sbs = []
        for j, sub_item in enumerate(item):
            if sub_item == 1.:
                sub_sbs.append(cfg.SBS_NAMES[j])
        gene_sbs.append(sub_sbs)

    # here , we find the intersections
    all_sbs_count = 49
    cancer_gene = {}

    cancer = cancer_type_sbs[k]
    for g, gene in enumerate(gene_sbs):
        # [ACC][gene1] = [SBS1,SBS3,...] ^ [SBS1,SBS3,...]
        cancer_gene[g] = len(list(set(cancer) & set(gene))) / all_sbs_count  # take the intersections

    # cancer gene : [{0:0,...186:0},{0:0,...186:0.2},....] of length 32
    # sort the gene in each cancer here

    # sorted_cancer_gene = [[(0,0.02),..(10,0) ],[(2,0.5),...(9,0)],[..],...] of length 32
    sorted_cancer_gene = dict(sorted(cancer_gene.items(), key=lambda item: item[1], reverse=True))

    # we find the top 5 mutated gene in each cancer here again
    gene_list_for_cancer = []
    gene_freq_list_for_cancer = []

    gene_list_final_for_cancer = []

    for gene in cfg.GENE_NAMES:
        gene_list_for_cancer.append((gene, cancer_prob[cancer_type][gene].values[0]))
        gene_freq_list_for_cancer.append(cancer_prob[cancer_type][gene].values[0])

    # find the top 5 gene's index in pandas frame
    top_5_index = list(reversed(
        sorted(range(len(gene_freq_list_for_cancer)), key=lambda i: gene_freq_list_for_cancer[i])[-5:]))

    # find those gene and their freq as (gene,freq)
    res_list = [gene_list_for_cancer[i] for i in top_5_index]

    # append the gene name into gene_list_final_for_cancer list
    # append the gene mutation frequency to gene_freq_list_final_for_cancer list
    for (a, b) in res_list:
        gene_list_final_for_cancer.append(a)

    gene_name_index_list = []

    for gene_weight in sorted_cancer_gene:
        gene_name_index_list.append(gene_weight)

    result_gene_name_list = [gene_list_final_for_cancer[i] for i in gene_name_index_list]

    result.append([cancer_type, ' '.join(result_gene_name_list)])

    # increasing the number that represent the cancer
    k += 1

# finally save the top 10 most related gene that has the greatest frequency in that cancer to the result.csv
df = pd.DataFrame(result)
df.to_csv('./result/ranking-result.csv', index=False, header=False)
