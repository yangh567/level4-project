import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# loading the cancer type sbs weight matrix from the running result
cancer_sbs_weight = np.load("../classification/result/cancer_type_normalized-weight.npy", mmap_mode='r')

# loading the gene sbs weight matrix from the running result
gene_sbs_weight = np.load("../classification/result/gene_normalized-weight.npy", mmap_mode='r')

# given the sbs columns
SBS_NAMES_lst = ['SBS4', 'SBS5', 'SBS1', 'SBS39', 'SBS36', 'SBS2', 'SBS13', 'SBS10b', 'SBS9', 'SBSPON', 'SBS3', 'SBS6',
                 'SBS30', 'SBSN', 'SBS10a', 'SBS15', 'SBS26', 'SBS29', 'SBS17b', 'SBS87', 'SBS16', 'SBS18', 'SBS52',
                 'SBS8', 'SBS7b', 'SBS40', 'SBS50', 'SBS24', 'SBS27', 'SBS42', 'SBS86', 'SBS57', 'SBS33', 'SBS90',
                 'SBS17a', 'SBS55', 'SBS22', 'SBS54', 'SBS48', 'SBS58', 'SBS28', 'SBS7a', 'SBS7d', 'SBS7c', 'SBS38',
                 'SBS84', 'SBS35', 'SBS14', 'SBS44']

# given the cancer labels
cancer_dict = {0: 'ACC', 1: 'BLCA', 2: 'BRCA', 3: 'CESC', 4: 'CHOL', 5: 'COAD', 6: 'DLBC', 7: 'ESCA', 8: 'GBM',
               9: 'HNSC', 10: 'KICH', 11: 'KIRC', 12: 'KIRP', 13: 'LAML', 14: 'LGG', 15: 'LIHC', 16: 'LUAD', 17: 'LUSC',
               18: 'MESO', 19: 'OV', 20: 'PAAD', 21: 'PCPG', 22: 'PRAD', 23: 'READ', 24: 'SKCM', 25: 'STAD', 26: 'TGCT',
               27: 'THCA', 28: 'THYM', 29: 'UCEC', 30: 'UCS', 31: 'UVM'}

# given the gene labels
gene_dict = {0: 'CCND1', 1: 'CCND2', 2: 'CCND3', 3: 'CCNE1', 4: 'CDK4', 5: 'CDK6', 6: 'E2F1', 7: 'E2F3', 8: 'YAP1',
             9: 'MYC', 10: 'MYCN', 11: 'ARRDC1', 12: 'KDM5A', 13: 'NFE2L2', 14: 'AKT1', 15: 'AKT2', 16: 'PIK3CA',
             17: 'PIK3CB', 18: 'PIK3R2', 19: 'RHEB', 20: 'RICTOR', 21: 'RPTOR', 22: 'EGFR', 23: 'ERBB2', 24: 'ERBB3',
             25: 'PDGFRA', 26: 'MET', 27: 'FGFR1', 28: 'FGFR2', 29: 'FGFR3', 30: 'FGFR4', 31: 'KIT', 32: 'IGF1R',
             33: 'KRAS', 34: 'HRAS', 35: 'BRAF', 36: 'RAF1', 37: 'RAC1', 38: 'MAPK1', 39: 'JAK2', 40: 'MDM2',
             41: 'MDM4', 42: 'CDKN1A', 43: 'CDKN1B', 44: 'CDKN2A', 45: 'CDKN2B', 46: 'CDKN2C', 47: 'RB1', 48: 'SAV1',
             49: 'LATS1', 50: 'LATS2', 51: 'PTPN14', 52: 'NF2', 53: 'FAT1', 54: 'MGA', 55: 'CNTN6', 56: 'CREBBP',
             57: 'EP300', 58: 'HES2', 59: 'HES3', 60: 'HES4', 61: 'HES5', 62: 'HEY1', 63: 'KAT2B', 64: 'NOTCH1',
             65: 'NOTCH2', 66: 'NOTCH3', 67: 'NOTCH4', 68: 'NOV', 69: 'PSEN2', 70: 'FBXW7', 71: 'NCOR1', 72: 'NCOR2',
             73: 'KEAP1', 74: 'CUL3', 75: 'INPP4B', 76: 'PIK3R1', 77: 'PTEN', 78: 'STK11', 79: 'TSC1', 80: 'TSC2',
             81: 'TGFBR1', 82: 'TGFBR2', 83: 'ACVR2A', 84: 'SMAD2', 85: 'SMAD3', 86: 'SMAD4', 87: 'NF1', 88: 'RASA1',
             89: 'CBL', 90: 'ERRFI1', 91: 'TP53', 92: 'ATM', 93: 'SFRP1', 94: 'ZNRF3', 95: 'AMER1', 96: 'APC',
             97: 'AXIN1', 98: 'DKK1', 99: 'DKK4', 100: 'RNF43', 101: 'TCF7L2', 102: 'ABL1', 103: 'ACVR1B', 104: 'AKT3',
             105: 'ALK', 106: 'ARAF', 107: 'AXIN2', 108: 'CDK2', 109: 'CHEK2', 110: 'CRB1', 111: 'CRB2', 112: 'CSNK1D',
             113: 'CSNK1E', 114: 'CTNNB1', 115: 'CUL1', 116: 'DCHS1', 117: 'DCHS2', 118: 'DKK2', 119: 'DKK3',
             120: 'DNER', 121: 'ERBB4', 122: 'ERF', 123: 'FAT2', 124: 'FAT3', 125: 'FAT4', 126: 'FLT3', 127: 'GRB2',
             128: 'GSK3B', 129: 'HDAC1', 130: 'HES1', 131: 'HEY2', 132: 'HEYL', 133: 'JAG2', 134: 'MAML3',
             135: 'MAP2K1', 136: 'MAP2K2', 137: 'MAX', 138: 'MLST8', 139: 'MLX', 140: 'MNT', 141: 'MOB1A', 142: 'MOB1B',
             143: 'MTOR', 144: 'MXI1', 145: 'NPRL2', 146: 'NPRL3', 147: 'NRAS', 148: 'NTRK1', 149: 'NTRK3',
             150: 'PIK3R3', 151: 'PLXNB1', 152: 'PPP2R1A', 153: 'PTPN11', 154: 'RET', 155: 'RIT1', 156: 'ROS1',
             157: 'RPS6KA3', 158: 'RPS6KB1', 159: 'SFRP2', 160: 'SFRP4', 161: 'SFRP5', 162: 'SOS1', 163: 'SOST',
             164: 'SPEN', 165: 'SPRED1', 166: 'STK3', 167: 'STK4', 168: 'TAOK1', 169: 'TAOK2', 170: 'TAOK3',
             171: 'TCF7', 172: 'TCF7L1', 173: 'TEAD2', 174: 'THBS2', 175: 'TLE1', 176: 'TLE2', 177: 'TLE3', 178: 'TLE4',
             179: 'WIF1', 180: 'WWC1', 181: 'CRB3', 182: 'LRP5', 183: 'NTRK2', 184: 'PDGFRB', 185: 'TEAD3',
             186: 'WWTR1'}

# assign the sbs signature columns to the cancer type sbs weight matrix,rename the index to cancer types
cancer_df = pd.DataFrame(cancer_sbs_weight, columns=SBS_NAMES_lst)
cancer_df.rename(index=cancer_dict, inplace=True)

# assign the sbs signature columns to the gene sbs weight matrix,rename the index to genes
gene_df = pd.DataFrame(gene_sbs_weight, columns=SBS_NAMES_lst)
gene_df.rename(index=gene_dict, inplace=True)

print(cancer_df)
print(gene_df)

# set up the heatmap for cancer type sbs weight matrix
plt.subplots(figsize=(20, 15))
sns.heatmap(cancer_df, annot=True, annot_kws={"size": 4}, cmap="RdYlGn")
plt.savefig('./figures/cancer_sbs_heatmap.png')

# set up the heatmap for gene sbs weight matrix
plt.subplots(figsize=(20, 15))
sns.heatmap(gene_df, annot=True, annot_kws={"size": 4}, cmap="RdYlGn")
plt.savefig('./figures/gene_sbs_heatmap.png')
