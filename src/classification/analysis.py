"""
The Analysis of cancer types and gene types
"""
import numpy as np
import pandas as pd
import my_config as cfg
from sklearn.preprocessing import MinMaxScaler

# load the file first
gene_path = './result/gene_type-weight_2.npy'
cancer_type_path = './result/cancer_type-weight_0.npy'


gene_weight = np.load(gene_path).T  # shape (187, 52)
cancer_type_weight = np.load(cancer_type_path).T

# standardize the weight
gene_scaler = MinMaxScaler()
cancer_type_scaler = MinMaxScaler()

gene_nor_weight = gene_scaler.fit_transform(gene_weight)
cancer_type_nor_weight = cancer_type_scaler.fit_transform(cancer_type_weight)

# normalize it to 0 and 1
cancer_type_zero_one_weight = cancer_type_nor_weight / np.sum(cancer_type_nor_weight, axis=0).reshape(1, 32)
gene_zero_one_weight = gene_nor_weight / np.sum(gene_nor_weight, axis=0).reshape(1, 187)

# save the data
np.save("./result/gene_normalized-weight.npy", gene_zero_one_weight)
np.save("./result/cancer_type_normalized-weight.npy", cancer_type_zero_one_weight)

# TODO output the suitable SBS sigantures
# calculate the mean of the weight and find the sbs signatures have the weight greater than mean
cancer_type_zero_one_weight = cancer_type_zero_one_weight.T
gene_zero_one_weight = gene_zero_one_weight.T
cancer_type_mean = np.std(cancer_type_zero_one_weight, axis=1)
for i, item in enumerate(cancer_type_zero_one_weight):
    item[item > cancer_type_mean[i]] = 1
    item[item <= cancer_type_mean[i]] = 0

gene_mean = np.std(gene_zero_one_weight, axis=1)
for i, item in enumerate(gene_zero_one_weight):
    item[item > gene_mean[i]] = 1
    item[item <= gene_mean[i]] = 0

# extract out the sbs signatures that have the weight as 1 in cancer types
cancer_type_sbs = []
gene_sbs = []

for i, item in enumerate(cancer_type_zero_one_weight):
    sub_sbs = []
    for j, sub_item in enumerate(item):
        if sub_item == 1.:
            sub_sbs.append(cfg.SBS_NAMES[j])
    cancer_type_sbs.append(sub_sbs)

# extract out the sbs signatures that have the weight as 1 in gene types
for i, item in enumerate(gene_zero_one_weight):
    sub_sbs = []
    for j, sub_item in enumerate(item):
        if sub_item == 1.:
            sub_sbs.append(cfg.SBS_NAMES[j])
    gene_sbs.append(sub_sbs)

# calculate the related gene of the cancer based on there intersection of sbs signatures
all_sbs_count = 49
cancer_gene = []

# i, cancer : ACC,[SBS1,SBS3...]
for i, cancer in enumerate(cancer_type_sbs):
    cancer_gene.append({})
    # j, gene : gene1,[SBS1,SBS3...]
    for j, gene in enumerate(gene_sbs):
        # cancer_gene[i].append([j, len(list(set(cancer) & set(gene))) / all_sbs_count])
        # [ACC][gene1] = [SBS1,SBS3,...] ^ [SBS1,SBS3,...]
        cancer_gene[i][j] = len(list(set(cancer) & set(gene))) / all_sbs_count  # take the intersections

# cancer gene : [{0:0,...186:0},{0:0,...186:0.2},....] of length 32
# sort the gene in each cancer here

# sorted_cancer_gene = [[(0,0.02),..(186,0) ],[(2,0.5),...(185,0)],[..],...] of length 32
sorted_cancer_gene = [sorted(item.items(), key=lambda kv: (kv[1], kv[0]), reverse=True) for item in cancer_gene]


# read the data here
# get the Occurrence probability of each gene in each cancer types

# like
#   gene1 gene2 gene3 gene4
#ACC 0.6  0.7   0.9   0.3

# we take those genes that are frequently mutated in each cancers by exam their patients

gene_prob = pd.read_csv('./result/gene_prob.csv')
cancer_prob = {}
for name, item in gene_prob.groupby('cancer type'):
    cancer_prob[name] = item

# cancer_prob = {'ACC':cancer type/ACC,CCND1/0.0,....;'BRAC':cancer type/BRAC,... }

# find the corresponding name of the gene for the corresponding cancer
result = [['cancer type', 'genes']]
for i, item in enumerate(sorted_cancer_gene):
    # 32 of them
    # (0,[(0,0.02),..(186,0) ])
    print(i, item)
    genes = []

    for gene in item:
        # 187 of them
        # (0,0.02),....(186,0)
        # gene[0] = 0, 186
        # only take those 10 genes that has their occurrence in that cancer > 0 and that gene's weight > 0 as well
        # if (cancer_prob[ACC][gene1].values[0] > 0 and gene[1] > 0)
        # (this is used to solve the problem of non-mutated gene have greater weight by extracting them)

        if cancer_prob[cfg.ORGAN_NAMES[i]][cfg.GENE_NAMES[gene[0]]].values[0] > 0.02 and gene[1] > 0:
            genes.append(cfg.GENE_NAMES[gene[0]])
        if len(genes) >= cfg.GENE_COUNT:
            break
    result.append([cfg.ORGAN_NAMES[i], ' '.join(genes)])

df = pd.DataFrame(result)
df.to_csv('./result/result.csv', index=False, header=False)

