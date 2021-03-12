"""

This file is used to draw the heatmap for display the
weight of sbs signatures in each cancer types or gene types

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

GENE_NAMES = ['CCND1', 'CCND2', 'CCND3', 'CCNE1', 'CDK4', 'CDK6', 'E2F1', 'E2F3', 'YAP1', 'MYC', 'MYCN', 'ARRDC1',
              'KDM5A',
              'NFE2L2', 'AKT1', 'AKT2', 'PIK3CA', 'PIK3CB', 'PIK3R2', 'RHEB', 'RICTOR', 'RPTOR', 'EGFR', 'ERBB2',
              'ERBB3',
              'PDGFRA', 'MET', 'FGFR1', 'FGFR2', 'FGFR3', 'FGFR4', 'KIT', 'IGF1R', 'KRAS', 'HRAS', 'BRAF', 'RAF1',
              'RAC1',
              'MAPK1', 'JAK2', 'MDM2', 'MDM4', 'CDKN1A', 'CDKN1B', 'CDKN2A', 'CDKN2B', 'CDKN2C', 'RB1', 'SAV1', 'LATS1',
              'LATS2',
              'PTPN14', 'NF2', 'FAT1', 'MGA', 'CNTN6', 'CREBBP', 'EP300', 'HES2', 'HES3', 'HES4', 'HES5', 'HEY1',
              'KAT2B', 'NOTCH1',
              'NOTCH2', 'NOTCH3', 'NOTCH4', 'NOV', 'PSEN2', 'FBXW7', 'NCOR1', 'NCOR2', 'KEAP1', 'CUL3', 'INPP4B',
              'PIK3R1', 'PTEN',
              'STK11', 'TSC1', 'TSC2', 'TGFBR1', 'TGFBR2', 'ACVR2A', 'SMAD2', 'SMAD3', 'SMAD4', 'NF1', 'RASA1', 'CBL',
              'ERRFI1',
              'TP53', 'ATM', 'SFRP1', 'ZNRF3', 'AMER1', 'APC', 'AXIN1', 'DKK1', 'DKK4', 'RNF43', 'TCF7L2', 'ABL1',
              'ACVR1B', 'AKT3',
              'ALK', 'ARAF', 'AXIN2', 'CDK2', 'CHEK2', 'CRB1', 'CRB2', 'CSNK1D', 'CSNK1E', 'CTNNB1', 'CUL1', 'DCHS1',
              'DCHS2', 'DKK2',
              'DKK3', 'DNER', 'ERBB4', 'ERF', 'FAT2', 'FAT3', 'FAT4', 'FLT3', 'GRB2', 'GSK3B', 'HDAC1', 'HES1', 'HEY2',
              'HEYL', 'JAG2',
              'MAML3', 'MAP2K1', 'MAP2K2', 'MAX', 'MLST8', 'MLX', 'MNT', 'MOB1A', 'MOB1B', 'MTOR', 'MXI1', 'NPRL2',
              'NPRL3', 'NRAS', 'NTRK1',
              'NTRK3', 'PIK3R3', 'PLXNB1', 'PPP2R1A', 'PTPN11', 'RET', 'RIT1', 'ROS1', 'RPS6KA3', 'RPS6KB1', 'SFRP2',
              'SFRP4', 'SFRP5',
              'SOS1', 'SOST', 'SPEN', 'SPRED1', 'STK3', 'STK4', 'TAOK1', 'TAOK2', 'TAOK3', 'TCF7', 'TCF7L1', 'TEAD2',
              'THBS2', 'TLE1',
              'TLE2', 'TLE3', 'TLE4', 'WIF1', 'WWC1', 'CRB3', 'LRP5', 'NTRK2', 'PDGFRB', 'TEAD3', 'WWTR1']

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
cancer_sbs_weight = np.load("../classification/result/cancer_type_normalized-weight.npy", mmap_mode='r').T

# assign the sbs signature columns to the cancer type sbs weight matrix,rename the index to cancer types
cancer_df = pd.DataFrame(cancer_sbs_weight, columns=SBS_NAMES_lst)
cancer_df.rename(index=cancer_dict, inplace=True)
print(cancer_df)

# set up the heatmap for cancer type sbs weight matrix
plt.subplots(figsize=(20, 15))
sns.heatmap(cancer_df, annot=True, annot_kws={"size": 4}, cmap='Reds')
plt.savefig('./figures/cancer_sbs_heatmap.png')
# ---------------------------------------------------------------------------------------------------------------------
gene_prob = pd.read_csv('../classification/result/gene_prob.csv')
cancer_prob = {}
for name, item in gene_prob.groupby('cancer type'):
    cancer_prob[name] = item
# ---------------------------------------------------------------------------------------------------------------------
# loading the cancer type sbs weight matrix from the running result
for cancer_type in cancer_list:
    # gene_list = []
    # gene_list_mutation_prob = []

    gene_sbs_weight_for_cancer = np.load(
        "../classification/result/gene_sbs_weights/gene_normalized_weights_for_each_cancer"
        "/gene_normalized-weight_" + cancer_type + ".npy", mmap_mode='r').T

    gene_list_for_cancer = []
    gene_freq_list_for_cancer = []

    gene_list_final_for_cancer = []

    for gene in GENE_NAMES:
        gene_list_for_cancer.append((gene, cancer_prob[cancer_type][gene].values[0]))
        gene_freq_list_for_cancer.append(cancer_prob[cancer_type][gene].values[0])

    # find the top 10 gene's index in pandas frame
    top_10_index = list(reversed(
        sorted(range(len(gene_freq_list_for_cancer)), key=lambda i: gene_freq_list_for_cancer[i])[-10:]))

    # find those gene and their freq as (gene,freq)
    res_list = [gene_list_for_cancer[i] for i in top_10_index]

    # append the gene name into gene_list_final_for_cancer list
    # append the gene mutation frequency to gene_freq_list_final_for_cancer list
    for (a, b) in res_list:
        gene_list_final_for_cancer.append(a)

    gene_dict = {}

    for i in range(len(gene_list_final_for_cancer)):
        gene_dict[i] = gene_list_final_for_cancer[i]

    gene_df = pd.DataFrame(gene_sbs_weight_for_cancer, columns=SBS_NAMES_lst)
    gene_df.rename(index=gene_dict, inplace=True)

    plt.subplots(figsize=(20, 15))
    sns.heatmap(gene_df, annot=True, annot_kws={"size": 4}, cmap="Reds")
    plt.savefig('./figures/gene_sbs_heatmaps_for_each_cancer/gene_sbs_heatmap_%s.png' % cancer_type)
