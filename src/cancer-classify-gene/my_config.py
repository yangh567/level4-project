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

CANCER_TYPES_NAMES = "Cancer.Types"

ORGAN_NAMES = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML',
               'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SKCM', 'STAD', 'TGCT', 'THCA',
               'THYM', 'UCEC', 'UCS', 'UVM']

SBS_NAMES = ['SBS4','SBS5','SBS1','SBS39','SBS36','SBS2','SBS13','SBS10b','SBS9','SBSPON','SBS3','SBS6','SBS30','SBSN','SBS10a','SBS15','SBS26','SBS29','SBS17b','SBS87','SBS16','SBS18','SBS52','SBS8','SBS7b','SBS40','SBS50','SBS24','SBS27','SBS42','SBS86','SBS57','SBS33','SBS90','SBS17a','SBS55','SBS22','SBS54','SBS48','SBS58','SBS28','SBS7a','SBS7d','SBS7c','SBS38','SBS84','SBS35','SBS14','SBS44']
ID = ['Sample_ID.1', 'Sample_ID']

