"""

This file is used to statistically summarize number
of mutation of the each gene in each cancer
and express it in probability form

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 2})
import warnings

figure_data = './small_data'

GENE_NAMES = ['CCND1', 'CCND2', 'CCND3', 'CCNE1', 'CDK4', 'CDK6', 'E2F1', 'E2F3', 'YAP1', 'MYC', 'MYCN', 'ARRDC1',
              'KDM5A', 'NFE2L2', 'AKT1', 'AKT2', 'PIK3CA', 'PIK3CB', 'PIK3R2', 'RHEB', 'RICTOR', 'RPTOR', 'EGFR',
              'ERBB2', 'ERBB3', 'PDGFRA', 'MET', 'FGFR1', 'FGFR2', 'FGFR3', 'FGFR4', 'KIT', 'IGF1R', 'KRAS', 'HRAS',
              'BRAF', 'RAF1', 'RAC1', 'MAPK1', 'JAK2', 'MDM2', 'MDM4', 'CDKN1A', 'CDKN1B', 'CDKN2A', 'CDKN2B', 'CDKN2C',
              'RB1', 'SAV1', 'LATS1', 'LATS2', 'PTPN14', 'NF2', 'FAT1', 'MGA', 'CNTN6', 'CREBBP', 'EP300', 'HES2',
              'HES3', 'HES4', 'HES5', 'HEY1', 'KAT2B', 'NOTCH1', 'NOTCH2', 'NOTCH3', 'NOTCH4', 'NOV', 'PSEN2', 'FBXW7',
              'NCOR1', 'NCOR2', 'KEAP1', 'CUL3', 'INPP4B', 'PIK3R1', 'PTEN', 'STK11', 'TSC1', 'TSC2', 'TGFBR1',
              'TGFBR2',
              'ACVR2A', 'SMAD2', 'SMAD3', 'SMAD4', 'NF1', 'RASA1', 'CBL', 'ERRFI1', 'TP53', 'ATM', 'SFRP1', 'ZNRF3',
              'AMER1', 'APC', 'AXIN1', 'DKK1', 'DKK4', 'RNF43', 'TCF7L2', 'ABL1', 'ACVR1B', 'AKT3', 'ALK', 'ARAF',
              'AXIN2', 'CDK2', 'CHEK2', 'CRB1', 'CRB2', 'CSNK1D', 'CSNK1E', 'CTNNB1', 'CUL1', 'DCHS1', 'DCHS2', 'DKK2',
              'DKK3', 'DNER', 'ERBB4', 'ERF', 'FAT2', 'FAT3', 'FAT4', 'FLT3', 'GRB2', 'GSK3B', 'HDAC1', 'HES1', 'HEY2',
              'HEYL', 'JAG2', 'MAML3', 'MAP2K1', 'MAP2K2', 'MAX', 'MLST8', 'MLX', 'MNT', 'MOB1A', 'MOB1B', 'MTOR',
              'MXI1', 'NPRL2', 'NPRL3', 'NRAS', 'NTRK1', 'NTRK3', 'PIK3R3', 'PLXNB1', 'PPP2R1A', 'PTPN11', 'RET',
              'RIT1',
              'ROS1', 'RPS6KA3', 'RPS6KB1', 'SFRP2', 'SFRP4', 'SFRP5', 'SOS1', 'SOST', 'SPEN', 'SPRED1', 'STK3', 'STK4',
              'TAOK1', 'TAOK2', 'TAOK3', 'TCF7', 'TCF7L1', 'TEAD2', 'THBS2', 'TLE1', 'TLE2', 'TLE3', 'TLE4', 'WIF1',
              'WWC1', 'CRB3', 'LRP5', 'NTRK2', 'PDGFRB', 'TEAD3', 'WWTR1']

warnings.filterwarnings('ignore')


def plt_figure(data, title):
    plt.title(title)
    plt.bar(x=data.keys(), height=data.values())
    for a, b in zip(data.keys(), data.values()):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=7)

    plt.xticks(rotation=70)
    plt.savefig(os.path.join(figure_data, '%s.png' % title))
    plt.close('all')


# read the data
o_data = pd.read_csv("small_data/small_data.csv")
result = {}
data = []

# name,item : ACC , sample 1 : gene1/0,gene2/0,gene3/1
#                 , sample 2 : gene1/0,gene2/0,gene2/1
for name, item in o_data.groupby('organ'):
    result[name] = {}
    # count will count all of the mutation in each of the patients in that name(cancer)
    count = item.shape[0]
    data.append([name])

    # if the gene has mutated 3 times we set its mutation as 1
    # we calculate the probability of occurrence frequency of each gene in each cancers here
    for gene in GENE_NAMES:
        item[gene][item[gene] >= 1] = 1
        item[gene][item[gene] < 1] = 0
        result[name][gene] = np.sum(item[gene] / count)
        # print(name, gene, result[name][gene])
        data[-1].append(result[name][gene])
df = pd.DataFrame(data)
df.columns = ['cancer type'] + GENE_NAMES
df.to_csv('../classification/result/gene_prob.csv', index=False)
print("The gene probability distribution is generated !")

# Draw the gene distribution in each cancer(DEPLICATED)

# for i in data:
#   title = i[0]
#  list_gene = i[1:]
# gene_dict = {}
# for j in range(len(list_gene)):
#   gene_dict[GENE_NAMES[j]] = j
# plt_figure(gene_dict,title)
