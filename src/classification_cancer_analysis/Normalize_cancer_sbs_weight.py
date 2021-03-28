"""
The normalization of weights of sbs signatures in cancer types
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('..','my_utilities')))
import numpy as np
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

