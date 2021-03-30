'''

This file is used to summarize the driver gene in each cancer we needed

'''

import pandas as pd
import numpy as np

ORGAN_NAMES = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COADREAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP',
               'LAML',
               'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'SARC', 'SKCM', 'TGCT',
               'THCA',
               'THYM', 'UCEC', 'UCS', 'UVM']

data = pd.read_csv("driver-gene-in-cancers.csv")

# collect the driver gene in each cancer
cancer_driver_gene_dict = {}
for cancer in ORGAN_NAMES:
    cancer_driver_gene_dict[cancer] = list(data[data["Cancer type"] == cancer]["Gene"].values)

print(cancer_driver_gene_dict)

# concatenate them to one list
driver_gene_list = []
for cancer in ORGAN_NAMES:
    for gene in cancer_driver_gene_dict[cancer]:
        driver_gene_list.append(gene)

# extract out all of the repeated gene
driver_gene_list = list(set(driver_gene_list))

# show me the length and content of gene
print(len(driver_gene_list))
print(driver_gene_list)