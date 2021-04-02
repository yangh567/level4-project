"""

This file is used to statistically summarize number
of mutation of the each gene in each cancer
and express it in probability form

"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('..', 'my_utilities')))
from my_utilities import my_config as cfg

matplotlib.rcParams.update({'font.size': 2})
import warnings

figure_data = './small_data'
warnings.filterwarnings('ignore')


# read the data
o_data = pd.read_csv("small_data/small_data3.csv")
result = {}
data = []

# name,item : ACC , sample 1 : gene1/0,gene2/0,gene3/1
#                 , sample 2 : gene1/0,gene2/0,gene2/1
for name, item in o_data.groupby('organ'):
    result[name] = {}
    # count will count all of the mutation in each of the patients in that name(cancer)
    count = item.shape[0]
    data.append([name])

    # if the gene has mutated multiple times or one time we set its mutation as 1
    # we calculate the probability of occurrence frequency of each gene in each cancers here
    for gene in cfg.GENE_NAMES:
        item[gene][item[gene] >= 1] = 1
        item[gene][item[gene] < 1] = 0
        result[name][gene] = np.sum(item[gene] / count)
        # print(name, gene, result[name][gene])
        data[-1].append(result[name][gene])
df = pd.DataFrame(data)
df.columns = ['cancer type'] + cfg.GENE_NAMES
df.to_csv('gene_distribution/gene_prob.csv', index=False)
print("The gene probability distribution is generated !")
