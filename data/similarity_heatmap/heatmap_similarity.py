"""


This code will draw the similarity between different sbs signatures and signatures extracted using NMF


"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ORGAN_NAMES = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP',
               'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC','SKCM',
               'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']


for cancer in ORGAN_NAMES:

    # loading the cancer bias from the result
    cancer_sbs_sim = pd.read_csv('../processed/sim.'+cancer+'.csv')

    first_column = cancer_sbs_sim.columns[0]

    sbs_list = list(cancer_sbs_sim[first_column])

    # Delete first
    df = cancer_sbs_sim.drop([first_column], axis=1)
    df.columns = [''] * len(df.columns)

    sbs_dict = {}
    for i, sig in enumerate(sbs_list):
        sbs_dict[i] = sig

    sig_list = list(set(cancer_sbs_sim.columns.tolist()) - {'Unnamed: 0'})

    cancer_df = pd.DataFrame(np.array(df), columns=sig_list)
    cancer_df.rename(index=sbs_dict, inplace=True)

    plt.subplots(figsize=(20, 15))
    sns.heatmap(cancer_df, annot=True, annot_kws={"size": 4}, cmap="RdYlGn")
    plt.savefig('../processed/class_graphs/class.'+cancer+'.png')
    plt.close()