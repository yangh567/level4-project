"""

This file is used to draw the heatmap for display the
weight of sbs signatures in each cancer types or gene types

"""
import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('..', 'my_utilities')))
from my_utilities import my_config as cfg

# given the cancer lists
cancer_list = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP',
               'LAML',
               'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'TGCT',
               'THCA',
               'THYM', 'UCEC', 'UCS', 'UVM']

# given the cancer labels
cancer_dict = {0: 'ACC', 1: 'BLCA', 2: 'BRCA', 3: 'CESC', 4: 'CHOL', 5: 'COAD', 6: 'DLBC', 7: 'ESCA', 8: 'GBM',
               9: 'HNSC', 10: 'KICH', 11: 'KIRC', 12: 'KIRP', 13: 'LAML', 14: 'LGG', 15: 'LIHC', 16: 'LUAD', 17: 'LUSC',
               18: 'MESO', 19: 'OV', 20: 'PAAD', 21: 'PCPG', 22: 'PRAD', 23: 'READ', 24: 'SARC', 25: 'SKCM', 26: 'TGCT',
               27: 'THCA', 28: 'THYM', 29: 'UCEC', 30: 'UCS', 31: 'UVM'}

# given the gene name columns
GENE_NAMES = ['MEN1', 'ATRX', 'CTNNB1', 'TP53', 'PRKAR1A', 'RXRA', 'CUL1', 'NFE2L2', 'STAG2', 'EP300', 'FAT1', 'FOXA1',
              'ATM',
              'ELF3', 'FOXQ1', 'ARID1A', 'ERBB3', 'PIK3CA', 'ASXL2', 'KRAS', 'TXNIP', 'ZFP36L1', 'HRAS', 'RHOB',
              'CREBBP',
              'NRAS', 'ERBB2', 'KANSL1', 'KDM6A', 'PSIP1', 'FGFR3', 'CDKN2A', 'SF1', 'GNA13', 'ERCC2', 'DIAPH2',
              'SF3B1',
              'PTEN', 'RB1', 'KMT2D', 'CDKN1A', 'RHOA', 'TSC1', 'SPTAN1', 'KLF5', 'RBM10', 'FBXW7', 'KMT2C', 'TBX3',
              'CDKN1B',
              'NF1', 'PTPRD', 'CBFB', 'CHD4', 'PIK3R1', 'GATA3', 'CTCF', 'MAP3K1', 'AKT1', 'MAP2K4', 'CDH1', 'GPS2',
              'CASP8',
              'NCOR1', 'BRCA1', 'RUNX1', 'HLA-B', 'SMAD4', 'NOTCH1', 'POLRMT', 'STK11', 'TGFBR2', 'MAPK1', 'LATS1',
              'IDH1', 'EPHA2',
              'PBRM1', 'BAP1', 'TGIF1', 'APC', 'ACVR2A', 'AMER1', 'PCBP1', 'BRAF', 'SOX9', 'TCF7L2', 'ZFP36L2', 'SMAD2',
              'GNAS', 'CARD11',
              'CD70', 'TNFAIP3', 'TMSB4X', 'BTG2', 'HIST1H1E', 'PIM1', 'CD79B', 'MYD88', 'B2M', 'NSD1', 'PTCH1',
              'ZNF750', 'TCF12', 'EGFR',
              'LZTR1', 'SPTA1', 'KEL', 'PDGFRA', 'GABRA6', 'CYLD', 'MYH9', 'FLNA', 'HUWE1', 'AJUBA', 'KEAP1', 'RAC1',
              'ARID2', 'CUL3', 'RASA1',
              'HLA-A', 'SETD2', 'MTOR', 'VHL', 'KDM5C', 'KIF1A', 'TCEB1', 'SMARCB1', 'MET', 'NF2', 'SMC1A', 'KIT',
              'U2AF1', 'PTPDC1', 'NPM1',
              'DNMT3A', 'PTPN11', 'ASXL1', 'IDH2', 'FLT3', 'WT1', 'CIC', 'ZBTB20', 'FUBP1', 'NIPBL', 'MAX', 'SMARCA4',
              'ZCCHC12', 'PLCG1',
              'NUP133', 'ALB', 'BRD7', 'APOB', 'TSC2', 'AXIN1', 'XPO1', 'WHSC1', 'CREB3L3', 'IL6ST', 'DHX9', 'RPS6KA3',
              'EEF1A1', 'MGA',
              'RIT1', 'FGFR2', 'ARHGAP35', 'LATS2', 'CDK12', 'ZNF133', 'RNF43', 'EEF2', 'RET', 'CSDE1', 'EPAS1',
              'MED12', 'ZMYM3', 'SPOP',
              'DACH1', 'COL5A1', 'PPP6C', 'MAP2K1', 'CDK4', 'RQCD1', 'MECOM', 'DDX3X', 'GNA11', 'PTMA', 'PPM1D',
              'EIF1AX', 'NUP93', 'GTF2I',
              'CHD3', 'MSH6', 'BCOR', 'ARID5B', 'CCND1', 'ACVR1', 'PDS5B', 'DICER1', 'SIN3A', 'MYCN', 'ZFHX3', 'FOXA2',
              'ATR', 'KMT2B',
              'PIK3R2', 'TAF1', 'RPL22', 'PPP2R1A', 'ZMYM2', 'SCAF4', 'INPPL1', 'CTNND1', 'SOS1', 'RRAS2', 'SOX17',
              'ATF7IP', 'RFC1',
              'BCL2L11', 'MAP3K4', 'ZBTB7B', 'PLCB4', 'GNAQ', 'SRSF2', 'CYSLTR2']

# given the sbs columns
SBS_NAMES_lst = ['SBS4', 'SBS5', 'SBS1', 'SBS39', 'SBS36', 'SBS2', 'SBS13', 'SBS10b', 'SBS9', 'SBSPON', 'SBS3', 'SBS6',
                 'SBS30',
                 'SBSN', 'SBS10a', 'SBS15', 'SBS26', 'SBS29', 'SBS17b', 'SBS87', 'SBS16', 'SBS18', 'SBS52', 'SBS8',
                 'SBS7b', 'SBS40',
                 'SBS50', 'SBS24', 'SBS27', 'SBS42', 'SBS86', 'SBS57', 'SBS33', 'SBS90', 'SBS17a', 'SBS55', 'SBS22',
                 'SBS54', 'SBS48',
                 'SBS58', 'SBS28', 'SBS7a', 'SBS7d', 'SBS7c', 'SBS38', 'SBS84', 'SBS35', 'SBS14', 'SBS44']
# ---------------------------------------------------------------------------------------------------------------------
# loading the cancer type sbs weight matrix from the running result
cancer_sbs_weight = np.load("./result/cancer_type_normalized-weight.npy", mmap_mode='r').T

# assign the sbs signature columns to the cancer type sbs weight matrix,rename the index to cancer types
cancer_df = pd.DataFrame(cancer_sbs_weight, columns=SBS_NAMES_lst)
cancer_df.rename(index=cancer_dict, inplace=True)
print(cancer_df)

# set up the heatmap for cancer type sbs weight matrix
plt.subplots(figsize=(20, 15))
sns.heatmap(cancer_df, annot=True, annot_kws={"size": 4}, cmap='Reds')
plt.savefig('./result/figures/cancer_sbs_heatmap.png')
# ---------------------------------------------------------------------------------------------------------------------
gene_prob = pd.read_csv('../statistics/gene_distribution/gene_prob.csv')
cancer_prob = {}
for name, item in gene_prob.groupby('cancer type'):
    cancer_prob[name] = item
# ---------------------------------------------------------------------------------------------------------------------
# loading the cancer type sbs weight matrix from the running result
for cancer_type in range(len(cancer_list)):

    # load the normalized weights for gene classification
    gene_sbs_weight_for_cancer = np.load(
        "./result/gene_sbs_weights/gene_normalized_weights_for_each_cancer"
        "/gene_normalized-weight_" + cancer_list[cancer_type] + ".npy", mmap_mode='r').T

    gene_list_for_cancer = []
    gene_freq_list_for_cancer = []

    gene_list_final_for_cancer = []

    for gene in GENE_NAMES:
        gene_list_for_cancer.append((gene, cancer_prob[cancer_list[cancer_type]][gene].values[0]))
        gene_freq_list_for_cancer.append(cancer_prob[cancer_list[cancer_type]][gene].values[0])

    # find the top 5 gene's index in pandas frame
    top_10_index = list(reversed(
        sorted(range(len(gene_freq_list_for_cancer)), key=lambda i: gene_freq_list_for_cancer[i])[-5:]))

    # find those gene and their freq as (gene,freq)
    res_list = [gene_list_for_cancer[i] for i in top_10_index]

    # append the gene name into gene_list_final_for_cancer list
    # append the gene mutation frequency to gene_freq_list_final_for_cancer list
    for (a, b) in res_list:
        gene_list_final_for_cancer.append(a)

    gene_dict = {}

    for i in range(len(gene_list_final_for_cancer)):
        gene_dict[i] = gene_list_final_for_cancer[i]

    # find the most weighted sbs names in that fold used as features for gene classification_cancer_gene_analysis
    # here we only investigate on 0 fold
    cancer_type_path = '../classification_cancer_gene_analysis/result/cancer_type-weight_4.npy'
    cancer_type_weight = np.load(cancer_type_path).T  # shape (10,32)
    cancer_type_scaler = MinMaxScaler()
    cancer_type_nor_weight = cancer_type_scaler.fit_transform(cancer_type_weight)
    # normalize it to 0 and 1
    cancer_type_zero_one_weight = cancer_type_nor_weight / np.sum(cancer_type_nor_weight, axis=0).reshape(1, 32)

    cancer_type_zero_one_weight_c = list(cancer_type_zero_one_weight[:, cancer_type])

    # we find the top 10 weighted sbs signatures comes handy in identify this cancer

    top_10_cancer_sbs_index = list(reversed(
        sorted(range(len(cancer_type_zero_one_weight_c)), key=lambda k: cancer_type_zero_one_weight_c[k])[
        -10:]))

    # save the result sbs columns to the list to help with visualization
    res_cancer_sbs_weight_list = [cfg.SBS_NAMES[s] for s in top_10_cancer_sbs_index]

    gene_df = pd.DataFrame(gene_sbs_weight_for_cancer, columns=res_cancer_sbs_weight_list)
    gene_df.rename(index=gene_dict, inplace=True)

    # visualize the result
    plt.subplots(figsize=(20, 15))
    sns.heatmap(gene_df, annot=True, annot_kws={"size": 4}, cmap="Reds")
    plt.savefig('./result/figures/gene_sbs_heatmaps_for_each_cancer/gene_sbs_heatmap_%s.png' % cancer_list[cancer_type])
    plt.close()
