"""

This file is used to provide the hyper parameters needed

training episodes : EPOCH
learning rate : LEARNING_RATE
mini_batch size : BATCH_SIZE
gene names : GENE_NAMES
sbs signatures : SBS_NAMES
cancer types : ORGAN_NAMES

data path : DATA_PATH
result saving path : C_V_DATA_PATH
fold numbers : CROSS_VALIDATION_COUNT
top n genes related to cancer : GENE_COUNT

"""
# train the hyper parameters

import pandas as pd

EPOCH = 10
LEARNING_RATE = 1e-2
BATCH_SIZE = 16

DATA_PATH = '../../data/processed/sample_id.sbs.organ.csv'
C_V_DATA_PATH = '../../data/cross_valid'

CROSS_VALIDATION_COUNT = 6
GENE_PROB = 0.
GENE_COUNT = 10


GENE_NAMES = ['CCND1', 'CCND2', 'CCND3', 'CCNE1', 'CDK4', 'CDK6', 'E2F1', 'E2F3', 'YAP1', 'MYC', 'MYCN', 'ARRDC1','KDM5A',
              'NFE2L2', 'AKT1', 'AKT2', 'PIK3CA', 'PIK3CB', 'PIK3R2', 'RHEB', 'RICTOR', 'RPTOR', 'EGFR', 'ERBB2','ERBB3',
              'PDGFRA', 'MET', 'FGFR1', 'FGFR2', 'FGFR3', 'FGFR4', 'KIT', 'IGF1R', 'KRAS', 'HRAS', 'BRAF', 'RAF1','RAC1',
              'MAPK1', 'JAK2', 'MDM2', 'MDM4', 'CDKN1A', 'CDKN1B', 'CDKN2A', 'CDKN2B', 'CDKN2C', 'RB1', 'SAV1', 'LATS1','LATS2',
              'PTPN14', 'NF2', 'FAT1', 'MGA', 'CNTN6', 'CREBBP', 'EP300', 'HES2', 'HES3', 'HES4', 'HES5', 'HEY1','KAT2B', 'NOTCH1',
              'NOTCH2', 'NOTCH3', 'NOTCH4', 'NOV', 'PSEN2', 'FBXW7', 'NCOR1', 'NCOR2', 'KEAP1', 'CUL3','INPP4B', 'PIK3R1', 'PTEN',
              'STK11', 'TSC1', 'TSC2', 'TGFBR1', 'TGFBR2', 'ACVR2A', 'SMAD2', 'SMAD3', 'SMAD4','NF1', 'RASA1', 'CBL', 'ERRFI1',
              'TP53', 'ATM', 'SFRP1', 'ZNRF3', 'AMER1', 'APC', 'AXIN1', 'DKK1', 'DKK4', 'RNF43','TCF7L2', 'ABL1', 'ACVR1B', 'AKT3',
              'ALK', 'ARAF', 'AXIN2', 'CDK2', 'CHEK2', 'CRB1', 'CRB2', 'CSNK1D', 'CSNK1E','CTNNB1', 'CUL1', 'DCHS1', 'DCHS2', 'DKK2',
              'DKK3', 'DNER', 'ERBB4', 'ERF', 'FAT2', 'FAT3', 'FAT4', 'FLT3', 'GRB2','GSK3B', 'HDAC1', 'HES1', 'HEY2', 'HEYL', 'JAG2',
              'MAML3', 'MAP2K1', 'MAP2K2', 'MAX', 'MLST8', 'MLX', 'MNT','MOB1A', 'MOB1B', 'MTOR', 'MXI1', 'NPRL2', 'NPRL3', 'NRAS', 'NTRK1',
              'NTRK3', 'PIK3R3', 'PLXNB1', 'PPP2R1A','PTPN11', 'RET', 'RIT1', 'ROS1', 'RPS6KA3', 'RPS6KB1', 'SFRP2', 'SFRP4', 'SFRP5',
              'SOS1', 'SOST', 'SPEN', 'SPRED1','STK3', 'STK4', 'TAOK1', 'TAOK2', 'TAOK3', 'TCF7', 'TCF7L1', 'TEAD2', 'THBS2', 'TLE1',
              'TLE2', 'TLE3', 'TLE4','WIF1', 'WWC1', 'CRB3', 'LRP5', 'NTRK2', 'PDGFRB', 'TEAD3', 'WWTR1']

CANCER_TYPES_NAMES = "Cancer.Types"

ORGAN_NAMES = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML',
               'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SKCM', 'STAD', 'TGCT', 'THCA',
               'THYM', 'UCEC', 'UCS', 'UVM']

SBS_NAMES = ['SBS4','SBS5','SBS1','SBS39','SBS36','SBS2','SBS13','SBS10b','SBS9','SBSPON','SBS3','SBS6','SBS30','SBSN','SBS10a','SBS15','SBS26','SBS29','SBS17b','SBS87','SBS16','SBS18','SBS52','SBS8','SBS7b','SBS40','SBS50','SBS24','SBS27','SBS42','SBS86','SBS57','SBS33','SBS90','SBS17a','SBS55','SBS22','SBS54','SBS48','SBS58','SBS28','SBS7a','SBS7d','SBS7c','SBS38','SBS84','SBS35','SBS14','SBS44']
ID = ['Sample_ID.1', 'Sample_ID']

