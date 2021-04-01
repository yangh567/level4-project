"""
The normalization of sbs weights in BLCA and LGG
"""
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('..','my_utilities')))

from sklearn.preprocessing import MinMaxScaler

# load the file first
cancer_type_path = './result/cancer_type-weight_3.npy'

cancer_type_weight = np.load(cancer_type_path).T

# standardize the weight
cancer_type_scaler = MinMaxScaler()

cancer_type_nor_weight = cancer_type_scaler.fit_transform(cancer_type_weight)

# normalize it to 0 and 1
cancer_type_zero_one_weight = cancer_type_nor_weight / np.sum(cancer_type_nor_weight, axis=0).reshape(1, 2)

# save the data
np.save("./result/cancer_type_normalized-weight.npy", cancer_type_zero_one_weight)
