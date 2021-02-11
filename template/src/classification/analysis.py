"""
The analysis of the relationship between cancer types and gene mutation status
"""
import numpy as np
import pandas as pd
import my_config as cfg
from sklearn.preprocessing import MinMaxScaler


# standardize the weights
gene_path = './result/gene_type-weight.npy'
cancer_type_path = './result/cancer_type-weight.npy'

gene_weight = np.load(gene_path).T    # shape (187, 52)
cancer_type_weight = np.load(cancer_type_path).T

# normalize the weight
gene_scaler = MinMaxScaler()
cancer_type_scaler = MinMaxScaler()

gene_nor_weight = gene_scaler.fit_transform(gene_weight).T
cancer_type_nor_weight = cancer_type_scaler.fit_transform(cancer_type_weight).T


# normalization
cancer_type_zero_one_weight = cancer_type_nor_weight / np.sum(cancer_type_nor_weight, axis=1).reshape(32, 1)
gene_zero_one_weight = gene_nor_weight / np.sum(gene_nor_weight, axis=1).reshape(187, 1)



# store the data

np.save("./result/gene_normalized-weight.npy", gene_zero_one_weight)
np.save("./result/cancer_type_normalized-weight.npy", cancer_type_zero_one_weight)

# TODO output the suitable SBS
# Statistically calculate mean, find the sbs weight for each cancer types greater than the mean and set it as 1,the rest will be 0
cancer_type_mean = np.std(cancer_type_zero_one_weight, axis=1)

for i, item in enumerate(cancer_type_zero_one_weight):
    item[item > cancer_type_mean[i]] = 1
    item[item <= cancer_type_mean[i]] = 0

# Statistically calculate mean, find the sbs weight for gene mutation combination greater than the mean and set it as 1,the rest will be 0
gene_mean = np.std(gene_zero_one_weight, axis=1)

for i, item in enumerate(gene_zero_one_weight):
    item[item > gene_mean[i]] = 1
    item[item <= gene_mean[i]] = 0



# extract all of the SBS
cancer_type_sbs = []
gene_sbs = []


for i, item in enumerate(cancer_type_zero_one_weight):
    sub_sbs = []
    for j, sub_item in enumerate(item):
        if sub_item == 1.:
            sub_sbs.append(cfg.SBS_NAMES[j])
    # print(len(sub_sbs))
    cancer_type_sbs.append(sub_sbs)




for i, item in enumerate(gene_zero_one_weight):
    sub_sbs = []
    for j, sub_item in enumerate(item):
        if sub_item == 1.:
            sub_sbs.append(cfg.SBS_NAMES[j])
    # print(len(sub_sbs))
    gene_sbs.append(sub_sbs)

pass


# find the gene related sbs for each cancer types
all_sbs_count = 52
cancer_gene = []


for i, cancer in enumerate(cancer_type_sbs):
    cancer_gene.append({})
    for j, gene in enumerate(gene_sbs):
        # cancer_gene[i].append([j, len(list(set(cancer) & set(gene))) / all_sbs_count])    # take Intersection
        cancer_gene[i][j] = len(list(set(cancer) & set(gene))) / all_sbs_count
    # Sort the genes, take the top 10
sorted_cancer_gene = [sorted(item.items(), key=lambda kv:(kv[1], kv[0]))[:10] for item in cancer_gene]

# find the corresponding cancer types and gene names
result = [['cancer type', 'genes']]
for i, item in enumerate(sorted_cancer_gene):
    result.append([cfg.ORGAN_NAMES[i], ' '.join([cfg.GENE_NAMES[sub_item[0]] for sub_item in item])])
df = pd.DataFrame(result)
df.to_csv('./result/result.csv', index=False, header=False)
