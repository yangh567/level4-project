"""
The normalization of sbs weights in BLCA and BRCA
"""
import numpy as np
import pandas as pd
import my_config as cfg
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
