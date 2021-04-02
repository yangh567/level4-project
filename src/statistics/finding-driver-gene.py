'''

This file is used to summarize the driver gene in each cancer we needed

'''
import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('..', 'my_utilities')))
from my_utilities import my_config as cfg


data = pd.read_csv("driver-gene-in-cancers.csv")

# collect the driver gene in each cancer
cancer_driver_gene_dict = {}
for cancer in cfg.ORGAN_NAMES:
    cancer_driver_gene_dict[cancer] = list(data[data["Cancer type"] == cancer]["Gene"].values)

print(cancer_driver_gene_dict)

# concatenate them to one list
driver_gene_list = []
for cancer in cfg.ORGAN_NAMES:
    for gene in cancer_driver_gene_dict[cancer]:
        driver_gene_list.append(gene)

# extract out all of the repeated gene
driver_gene_list = list(set(driver_gene_list))

# show me the length and content of gene
print(len(driver_gene_list))
print(driver_gene_list)