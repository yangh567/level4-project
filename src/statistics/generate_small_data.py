import pandas as pd

use_columns = ['organ', 'MEN1', 'ATRX', 'CTNNB1', 'TP53', 'PRKAR1A', 'RXRA', 'CUL1', 'NFE2L2', 'STAG2', 'EP300', 'FAT1',
               'FOXA1', 'ATM',
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
               'PBRM1', 'BAP1', 'TGIF1', 'APC', 'ACVR2A', 'AMER1', 'PCBP1', 'BRAF', 'SOX9', 'TCF7L2', 'ZFP36L2',
               'SMAD2', 'GNAS', 'CARD11',
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
# we only need to extract those gene and their caner type to calculate their occurrence in each type in all samples
data = pd.read_csv('../../data/cross_valid/validation_dataset.csv', usecols=use_columns, low_memory=True)

data = data.fillna(0)
print(data.columns.tolist())

data.to_csv('small_data/small_data3.csv')
