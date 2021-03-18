"""

This file is used to provide the hyper parameters needed

training episodes : EPOCH
learning rate : LEARNING_RATE
mini_batch size : BATCH_SIZE
gene names : GENE_NAMES
list of driver gene in each cancer : GENE_NAMES_DICT
sbs signatures : SBS_NAMES
cancer types : ORGAN_NAMES

data path : DATA_PATH
result saving path : C_V_DATA_PATH
fold numbers : CROSS_VALIDATION_COUNT
top n genes related to cancer : GENE_COUNT

"""
# train the hyper parameters

import pandas as pd

EPOCH = 1000
LEARNING_RATE = 1e-3
BATCH_SIZE = 64

DATA_PATH = '../../data/processed/sample_id.sbs.organ.csv'
C_V_DATA_PATH = '../../data/cross_valid'

CROSS_VALIDATION_COUNT = 6
GENE_PROB = 0.
GENE_COUNT = 10

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

GENE_NAMES_DICT = {'ACC': ['ATRX', 'CTNNB1', 'MEN1', 'PRKAR1A', 'TP53'],
                   'BLCA': ['ARID1A', 'ASXL2', 'ATM', 'CDKN1A', 'CDKN2A', 'CREBBP', 'CTNNB1', 'CUL1', 'DIAPH2', 'ELF3',
                            'EP300', 'ERBB2', 'ERBB3', 'ERCC2', 'FAT1', 'FBXW7', 'FGFR3', 'FOXA1', 'FOXQ1', 'GNA13',
                            'HRAS', 'KANSL1', 'KDM6A', 'KLF5', 'KMT2C', 'KMT2D', 'KRAS', 'NFE2L2', 'NRAS', 'PIK3CA',
                            'PSIP1', 'PTEN', 'RB1', 'RBM10', 'RHOA', 'RHOB', 'RXRA', 'SF1', 'SF3B1', 'SPTAN1', 'STAG2',
                            'TP53', 'TSC1', 'TXNIP', 'ZFP36L1'],
                   'BRCA': ['AKT1', 'ARID1A', 'BRCA1', 'CASP8', 'CBFB', 'CDH1', 'CDKN1B', 'CHD4', 'CTCF', 'ERBB2',
                            'FBXW7', 'FOXA1', 'GATA3', 'GPS2', 'KMT2C', 'KRAS', 'MAP2K4', 'MAP3K1', 'NCOR1', 'NF1',
                            'PIK3CA', 'PIK3R1', 'PTEN', 'PTPRD', 'RB1', 'RUNX1', 'SF3B1', 'TBX3', 'TP53'],
                   'CESC': ['AKT1', 'ARID1A', 'CASP8', 'EP300', 'ERBB2', 'ERBB3', 'FAT1', 'FBXW7', 'HLA-B', 'KLF5',
                            'KMT2C', 'KMT2D', 'KRAS', 'LATS1', 'MAP3K1', 'MAPK1', 'NFE2L2', 'NOTCH1', 'PIK3CA',
                            'POLRMT', 'PTEN', 'RB1', 'SMAD4', 'STK11', 'TGFBR2', 'TP53'],
                   'CHOL': ['ARID1A', 'BAP1', 'EPHA2', 'IDH1', 'PBRM1'],
                   'COAD': ['ACVR2A', 'AMER1', 'APC', 'ARID1A', 'BRAF', 'CTNNB1', 'FBXW7', 'GNAS', 'KRAS', 'NRAS',
                            'PCBP1', 'PIK3CA', 'PTEN', 'SMAD2', 'SMAD4', 'SOX9', 'TCF7L2', 'TGIF1', 'TP53', 'ZFP36L2'],
                   'READ': ['ACVR2A', 'AMER1', 'APC', 'ARID1A', 'BRAF', 'CTNNB1', 'FBXW7', 'GNAS', 'KRAS', 'NRAS',
                            'PCBP1', 'PIK3CA', 'PTEN', 'SMAD2', 'SMAD4', 'SOX9', 'TCF7L2', 'TGIF1', 'TP53', 'ZFP36L2'],
                   'DLBC': ['B2M', 'BTG2', 'CARD11', 'CD70', 'CD79B', 'HIST1H1E', 'HLA-B', 'KMT2D', 'MYD88', 'PIM1',
                            'TMSB4X', 'TNFAIP3', 'TP53'],
                   'ESCA': ['ARID1A', 'CDKN2A', 'ERBB2', 'FBXW7', 'KMT2D', 'NFE2L2', 'NOTCH1', 'NSD1', 'PIK3CA',
                            'PTCH1', 'SMAD4', 'TP53', 'ZNF750'],
                   'GBM': ['ATRX', 'BRAF', 'EGFR', 'GABRA6', 'IDH1', 'KEL', 'LZTR1', 'NF1', 'PDGFRA', 'PIK3CA',
                           'PIK3R1', 'PTEN', 'RB1', 'SPTA1', 'STAG2', 'TCF12', 'TP53'],
                   'HNSC': ['AJUBA', 'ARID2', 'CASP8', 'CDKN2A', 'CUL3', 'CYLD', 'EP300', 'EPHA2', 'FAT1', 'FBXW7',
                            'FGFR3', 'FLNA', 'HLA-A', 'HLA-B', 'HRAS', 'HUWE1', 'KDM6A', 'KEAP1', 'KMT2D', 'MAPK1',
                            'MYH9', 'NFE2L2', 'NOTCH1', 'NSD1', 'PIK3CA', 'PTEN', 'RAC1', 'RASA1', 'RB1', 'RHOA',
                            'TGFBR2', 'TP53', 'ZNF750'],
                   'KICH': ['PTEN', 'TP53'],
                   'KIRC': ['ATM', 'BAP1', 'KDM5C', 'KIF1A', 'MTOR', 'PBRM1', 'PIK3CA', 'PTEN', 'SETD2', 'TCEB1',
                            'TP53', 'VHL'],
                   'KIRP': ['BAP1', 'CUL3', 'KDM6A', 'KRAS', 'MET', 'NF2', 'PIK3CA', 'SETD2', 'SMARCB1'],
                   'LAML': ['ASXL1', 'DNMT3A', 'FLT3', 'IDH1', 'IDH2', 'KIT', 'KRAS', 'NPM1', 'PTPDC1', 'PTPN11',
                            'RUNX1', 'SF3B1', 'SMC1A', 'TP53', 'U2AF1', 'WT1'],
                   'LGG': ['ARID1A', 'ARID2', 'ATRX', 'CIC', 'EGFR', 'FUBP1', 'IDH1', 'IDH2', 'MAX', 'NF1', 'NIPBL',
                           'NOTCH1', 'NRAS', 'PIK3CA', 'PIK3R1', 'PLCG1', 'PTEN', 'PTPN11', 'SETD2', 'SMARCA4', 'TCF12',
                           'TP53', 'ZBTB20', 'ZCCHC12'],
                   'LIHC': ['ACVR2A', 'ALB', 'APOB', 'ARID1A', 'ARID2', 'AXIN1', 'BAP1', 'BRD7', 'CDKN1A', 'CDKN2A',
                            'CREB3L3', 'CTNNB1', 'DHX9', 'EEF1A1', 'GNAS', 'HIST1H1E', 'IDH1', 'IL6ST', 'KEAP1', 'KRAS',
                            'LZTR1', 'NFE2L2', 'NRAS', 'NUP133', 'PIK3CA', 'RB1', 'RPS6KA3', 'SMARCA4', 'TP53', 'TSC2',
                            'WHSC1', 'XPO1'],
                   'LUAD': ['ARID1A', 'ATM', 'BRAF', 'CDKN2A', 'CTNNB1', 'EGFR', 'KEAP1', 'KRAS', 'MET', 'MGA', 'NF1',
                            'PIK3CA', 'RB1', 'RBM10', 'RIT1', 'SETD2', 'SMARCA4', 'STK11', 'TP53', 'U2AF1'],
                   'LUSC': ['ARHGAP35', 'ARID1A', 'CDKN2A', 'CUL3', 'EP300', 'FAT1', 'FBXW7', 'FGFR2', 'HLA-A', 'HRAS',
                            'KDM6A', 'KEAP1', 'KLF5', 'KMT2D', 'NF1', 'NFE2L2', 'NOTCH1', 'PIK3CA', 'PTEN', 'RASA1',
                            'RB1', 'TP53'],
                   'MESO': ['BAP1', 'LATS2', 'NF2', 'SETD2', 'TP53'],
                   'OV': ['BRCA1', 'CDK12', 'KRAS', 'NF1', 'NRAS', 'RB1', 'TP53', 'ZNF133'],
                   'PAAD': ['ARID1A', 'CDKN2A', 'EEF2', 'GNAS', 'KDM6A', 'KRAS', 'RNF43', 'SMAD4', 'TGFBR2', 'TP53',
                            'U2AF1'],
                   'PCPG': ['CSDE1', 'EPAS1', 'HRAS', 'NF1', 'RET'],
                   'PRAD': ['AKT1', 'APC', 'BRAF', 'CTNNB1', 'FOXA1', 'HRAS', 'IDH1', 'KDM6A', 'KMT2D', 'MED12',
                            'PIK3CA', 'PTEN', 'SPOP', 'TP53', 'ZMYM3'],
                   'SARC': ['ATRX', 'RB1', 'TP53'],
                   'SKCM': ['ARID2', 'BRAF', 'BRD7', 'CDK4', 'CDKN2A', 'COL5A1', 'CTNNB1', 'DACH1', 'DDX3X', 'GNA11',
                            'HRAS', 'IDH1', 'KIT', 'KRAS', 'MAP2K1', 'MECOM', 'NF1', 'NRAS', 'PPP6C', 'PTEN', 'RAC1',
                            'RB1', 'RQCD1', 'TP53'],
                   'TGCT': ['KIT', 'KRAS', 'NRAS', 'PIK3CA', 'PTMA'],
                   'THCA': ['AKT1', 'BRAF', 'DNMT3A', 'EIF1AX', 'HRAS', 'KRAS', 'NRAS', 'NUP93', 'PPM1D'],
                   'THYM': ['GTF2I', 'HRAS', 'NRAS', 'TP53'],
                   'UCEC': ['ACVR1', 'AKT1', 'ARHGAP35', 'ARID1A', 'ARID5B', 'ATF7IP', 'ATM', 'ATR', 'BCOR', 'CCND1',
                            'CHD3', 'CHD4', 'CTCF', 'CTNNB1', 'CTNND1', 'DICER1', 'EP300', 'ERBB2', 'ERBB3', 'FAT1',
                            'FBXW7', 'FGFR2', 'FOXA2', 'INPPL1', 'KANSL1', 'KMT2B', 'KMT2C', 'KRAS', 'MAP3K1', 'MAX',
                            'MSH6', 'MYCN', 'NFE2L2', 'NRAS', 'PBRM1', 'PDS5B', 'PIK3CA', 'PIK3R1', 'PIK3R2', 'PPP2R1A',
                            'PTEN', 'RB1', 'RPL22', 'RRAS2', 'SCAF4', 'SIN3A', 'SMC1A', 'SOS1', 'SOX17', 'SPOP', 'TAF1',
                            'TP53', 'U2AF1', 'ZFHX3', 'ZMYM2'],
                   'UCS': ['ARID1A', 'BCL2L11', 'CHD4', 'FBXW7', 'KRAS', 'MAP3K4', 'PIK3CA', 'PIK3R1', 'PPP2R1A',
                           'PTEN', 'RB1', 'RFC1', 'TP53', 'ZBTB7B'],
                   'UVM': ['BAP1', 'CYSLTR2', 'EIF1AX', 'GNA11', 'GNAQ', 'PLCB4', 'SF3B1', 'SRSF2']}

ORGAN_NAMES = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP',
               'LAML',
               'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'TGCT',
               'THCA',
               'THYM', 'UCEC', 'UCS', 'UVM']

SBS_NAMES = ['SBS4', 'SBS5', 'SBS1', 'SBS39', 'SBS36', 'SBS2', 'SBS13', 'SBS10b', 'SBS9', 'SBSPON', 'SBS3', 'SBS6',
             'SBS30', 'SBSN', 'SBS10a', 'SBS15', 'SBS26', 'SBS29', 'SBS17b', 'SBS87', 'SBS16', 'SBS18', 'SBS52', 'SBS8',
             'SBS7b', 'SBS40', 'SBS50', 'SBS24', 'SBS27', 'SBS42', 'SBS86', 'SBS57', 'SBS33', 'SBS90', 'SBS17a',
             'SBS55', 'SBS22', 'SBS54', 'SBS48', 'SBS58', 'SBS28', 'SBS7a', 'SBS7d', 'SBS7c', 'SBS38', 'SBS84', 'SBS35',
             'SBS14', 'SBS44']
