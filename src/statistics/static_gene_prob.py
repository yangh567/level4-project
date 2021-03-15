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

GENE_NAMES = ['MEN1', 'ATRX', 'CTNNB1', 'TP53', 'PRKAR1A', 'RXRA', 'CUL1', 'NFE2L2', 'STAG2', 'EP300', 'FAT1', 'FOXA1', 'ATM',
              'ELF3', 'FOXQ1', 'ARID1A', 'ERBB3', 'PIK3CA', 'ASXL2', 'KRAS', 'TXNIP', 'ZFP36L1', 'HRAS', 'RHOB', 'CREBBP',
              'NRAS', 'ERBB2', 'KANSL1', 'KDM6A', 'PSIP1', 'FGFR3', 'CDKN2A', 'SF1', 'GNA13', 'ERCC2', 'DIAPH2', 'SF3B1',
              'PTEN', 'RB1', 'KMT2D', 'CDKN1A', 'RHOA', 'TSC1', 'SPTAN1', 'KLF5', 'RBM10', 'FBXW7', 'KMT2C', 'TBX3', 'CDKN1B',
              'NF1', 'PTPRD', 'CBFB', 'CHD4', 'PIK3R1', 'GATA3', 'CTCF', 'MAP3K1', 'AKT1', 'MAP2K4', 'CDH1', 'GPS2', 'CASP8',
              'NCOR1', 'BRCA1', 'RUNX1', 'HLA-B', 'SMAD4', 'NOTCH1', 'POLRMT', 'STK11', 'TGFBR2', 'MAPK1', 'LATS1', 'IDH1', 'EPHA2',
              'PBRM1', 'BAP1', 'TGIF1', 'APC', 'ACVR2A', 'AMER1', 'PCBP1', 'BRAF', 'SOX9', 'TCF7L2', 'ZFP36L2', 'SMAD2', 'GNAS', 'CARD11',
              'CD70', 'TNFAIP3', 'TMSB4X', 'BTG2', 'HIST1H1E', 'PIM1', 'CD79B', 'MYD88', 'B2M', 'NSD1', 'PTCH1', 'ZNF750', 'TCF12', 'EGFR',
              'LZTR1', 'SPTA1', 'KEL', 'PDGFRA', 'GABRA6', 'CYLD', 'MYH9', 'FLNA', 'HUWE1', 'AJUBA', 'KEAP1', 'RAC1', 'ARID2', 'CUL3', 'RASA1',
              'HLA-A', 'SETD2', 'MTOR', 'VHL', 'KDM5C', 'KIF1A', 'TCEB1', 'SMARCB1', 'MET', 'NF2', 'SMC1A', 'KIT', 'U2AF1', 'PTPDC1', 'NPM1',
              'DNMT3A', 'PTPN11', 'ASXL1', 'IDH2', 'FLT3', 'WT1', 'CIC', 'ZBTB20', 'FUBP1', 'NIPBL', 'MAX', 'SMARCA4', 'ZCCHC12', 'PLCG1',
              'NUP133', 'ALB', 'BRD7', 'APOB', 'TSC2', 'AXIN1', 'XPO1', 'WHSC1', 'CREB3L3', 'IL6ST', 'DHX9', 'RPS6KA3', 'EEF1A1', 'MGA',
              'RIT1', 'FGFR2', 'ARHGAP35', 'LATS2', 'CDK12', 'ZNF133', 'RNF43', 'EEF2', 'RET', 'CSDE1', 'EPAS1', 'MED12', 'ZMYM3', 'SPOP',
              'DACH1', 'COL5A1', 'PPP6C', 'MAP2K1', 'CDK4', 'RQCD1', 'MECOM', 'DDX3X', 'GNA11', 'PTMA', 'PPM1D', 'EIF1AX', 'NUP93', 'GTF2I',
              'CHD3', 'MSH6', 'BCOR', 'ARID5B', 'CCND1', 'ACVR1', 'PDS5B', 'DICER1', 'SIN3A', 'MYCN', 'ZFHX3', 'FOXA2', 'ATR', 'KMT2B',
              'PIK3R2', 'TAF1', 'RPL22', 'PPP2R1A', 'ZMYM2', 'SCAF4', 'INPPL1', 'CTNND1', 'SOS1', 'RRAS2', 'SOX17', 'ATF7IP', 'RFC1',
              'BCL2L11', 'MAP3K4', 'ZBTB7B', 'PLCB4', 'GNAQ', 'SRSF2', 'CYSLTR2']

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
df.to_csv('../classification-feature-extraction/result/gene_prob.csv', index=False)
print("The gene probability distribution is generated !")

# Draw the gene distribution in each cancer(DEPLICATED)

# for i in data:
#   title = i[0]
#  list_gene = i[1:]
# gene_dict = {}
# for j in range(len(list_gene)):
#   gene_dict[GENE_NAMES[j]] = j
# plt_figure(gene_dict,title)
