"""

This file is used to write 187 driver genes into a file for calculating their probabilities(DEPRECATED)

"""
gene_list = ['CCND1', 'CCND2', 'CCND3', 'CCNE1', 'CDK4', 'CDK6', 'E2F1', 'E2F3', 'YAP1', 'MYC', 'MYCN', 'ARRDC1',
             'KDM5A', 'NFE2L2', 'AKT1', 'AKT2', 'PIK3CA', 'PIK3CB', 'PIK3R2', 'RHEB', 'RICTOR', 'RPTOR', 'EGFR',
             'ERBB2', 'ERBB3', 'PDGFRA', 'MET', 'FGFR1', 'FGFR2', 'FGFR3', 'FGFR4', 'KIT', 'IGF1R', 'KRAS', 'HRAS',
             'BRAF', 'RAF1', 'RAC1', 'MAPK1', 'JAK2', 'MDM2', 'MDM4', 'CDKN1A', 'CDKN1B', 'CDKN2A', 'CDKN2B', 'CDKN2C',
             'RB1', 'SAV1', 'LATS1', 'LATS2', 'PTPN14', 'NF2', 'FAT1', 'MGA', 'CNTN6', 'CREBBP', 'EP300', 'HES2',
             'HES3', 'HES4', 'HES5', 'HEY1', 'KAT2B', 'NOTCH1', 'NOTCH2', 'NOTCH3', 'NOTCH4', 'NOV', 'PSEN2', 'FBXW7',
             'NCOR1', 'NCOR2', 'KEAP1', 'CUL3', 'INPP4B', 'PIK3R1', 'PTEN', 'STK11', 'TSC1', 'TSC2', 'TGFBR1', 'TGFBR2',
             'ACVR2A', 'SMAD2', 'SMAD3', 'SMAD4', 'NF1', 'RASA1', 'CBL', 'ERRFI1', 'TP53', 'ATM', 'SFRP1', 'ZNRF3',
             'AMER1', 'APC', 'AXIN1', 'DKK1', 'DKK4', 'RNF43', 'TCF7L2', 'ABL1', 'ACVR1B', 'AKT3', 'ALK', 'ARAF',
             'AXIN2', 'CDK2', 'CHEK2', 'CRB1', 'CRB2', 'CSNK1D', 'CSNK1E', 'CTNNB1', 'CUL1', 'DCHS1', 'DCHS2', 'DKK2',
             'DKK3', 'DNER', 'ERBB4', 'ERF', 'FAT2', 'FAT3', 'FAT4', 'FLT3', 'GRB2', 'GSK3B', 'HDAC1', 'HES1', 'HEY2',
             'HEYL', 'JAG2', 'MAML3', 'MAP2K1', 'MAP2K2', 'MAX', 'MLST8', 'MLX', 'MNT', 'MOB1A', 'MOB1B', 'MTOR',
             'MXI1', 'NPRL2', 'NPRL3', 'NRAS', 'NTRK1', 'NTRK3', 'PIK3R3', 'PLXNB1', 'PPP2R1A', 'PTPN11', 'RET', 'RIT1',
             'ROS1', 'RPS6KA3', 'RPS6KB1', 'SFRP2', 'SFRP4', 'SFRP5', 'SOS1', 'SOST', 'SPEN', 'SPRED1', 'STK3', 'STK4',
             'TAOK1', 'TAOK2', 'TAOK3', 'TCF7', 'TCF7L1', 'TEAD2', 'THBS2', 'TLE1', 'TLE2', 'TLE3', 'TLE4', 'WIF1',
             'WWC1', 'CRB3', 'LRP5', 'NTRK2', 'PDGFRB', 'TEAD3', 'WWTR1']

f = open("geneList.txt", "a")
for i in gene_list:
    f.write(i)
    f.write("\n")
f.close()
