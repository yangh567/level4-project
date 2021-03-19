"""
The Analysis of cancer types and gene types
"""
import os,sys
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('..','my_utilities')))
from my_utilities import my_config as cfg
from sklearn.preprocessing import MinMaxScaler

# load the file first

# The loading of cancer's sbs weights start from here

cancer_type_path = '../classification_cancer_gene_analysis/result/cancer_type-weight_4.npy'
cancer_type_weight = np.load(cancer_type_path).T  # shape (49,32)
cancer_type_scaler = MinMaxScaler()
cancer_type_nor_weight = cancer_type_scaler.fit_transform(cancer_type_weight)
# normalize it to 0 and 1
cancer_type_zero_one_weight = cancer_type_nor_weight / np.sum(cancer_type_nor_weight, axis=0).reshape(1, 32)
# save the data
np.save("./result/cancer_type_normalized-weight.npy", cancer_type_zero_one_weight)

# The loading of gene's sbs weights start from here
cancer_list = cfg.ORGAN_NAMES


result = [['cancer type', 'genes']]
# here we only investigate on 0 fold
for cancer_type in cancer_list:
    # used for constructing and saving result data frame later
    gene_path = './result/gene_sbs_weights/gene_type-weight_in_fold4_for_' + cancer_type + '.npy'

    gene_weight = np.load(gene_path).T  # shape (49, 10)

    # standardize the weight
    gene_scaler = MinMaxScaler()

    gene_nor_weight = gene_scaler.fit_transform(gene_weight)

    # normalize it to 0 and 1
    gene_zero_one_weight = gene_nor_weight / np.sum(gene_nor_weight, axis=0).reshape(1, 5)
    np.save(
        "./result/gene_sbs_weights/gene_normalized_weights_for_each_cancer/gene_normalized-weight_%s.npy" % cancer_type,
        gene_zero_one_weight)

