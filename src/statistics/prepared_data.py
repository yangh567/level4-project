"""

This file is used to split the original data into 6 folds,
5 fold used for 5-fold cross validation and one fold for
validation in each fold

"""

import os, sys
import random
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('..', 'my_utilities')))
from my_utilities import my_config as cfg
from my_utilities import my_tools as tool

import matplotlib.pyplot as plt

figure_data = 'cross_valid_static'

if not os.path.exists(figure_data):
    os.makedirs(figure_data)


# the function used to plot the cancer type distribution status in each of the fold
def plt_figure(data, title):
    plt.title(title)
    plt.bar(x=data.keys(), height=data.values())
    for a, b in zip(data.keys(), data.values()):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=6)

    plt.xticks(rotation=70)
    plt.savefig(os.path.join(figure_data, '%s.png' % title))
    plt.close('all')


def k_fold_split(train_df,gene_prob_in_cancer,k):
    # start the stratified sampling
    types = list(train_df.groupby("organ"))  # we group the data using cancer types for future sampling
    result = []
    indexs1 = []
    indexs0 = []

    # we set up the cancer type encoding here for the find_top_gene function to find top gene in that cancer
    cancer_type = 0
    # we generate the data from each of the cancer type and shuffle their index
    for cancer, data in types:
        # rest the index in each cancer
        index = list(range(len(data)))
        # shuffle the index
        np.random.shuffle(index)

        # append the index which have the driver gene not mutated
        index_cancer_0 = []
        # append the index which have the driver gene mutated
        index_cancer_1 = []

        for indx in index:
            gene, _, _ = tool.find_top_gene(cancer_type, gene_prob_in_cancer)
            # if the driver gene is not mutated
            if data.iloc[indx][gene][0] == 0:
                index_cancer_0.append(indx)
            # if the driver gene is mutated
            if data.iloc[indx][gene][0] == 1:
                index_cancer_1.append(indx)
        # only want to take k samples from each cancer and put into each fold
        indexs0.append((index_cancer_0, int(len(index_cancer_0) / k)))
        indexs1.append((index_cancer_1, int(len(index_cancer_1) / k)))

        # increment the cancer type
        cancer_type += 1

    for fold in range(k):
        tmp_result = []
        for i, type_ in enumerate(types):
            cancer, data = type_
            # find the number of samples we want to assign to each fold
            num_0 = indexs0[i][1]
            num_1 = indexs1[i][1]

            if fold + 1 < k:
                # take k samples that have the driver gene in that cancer not mutated
                index0 = indexs0[i][0][fold * num_0:(fold + 1) * num_0]
                # take k samples that have the driver gene in that cancer mutated
                index1 = indexs1[i][0][fold * num_1:(fold + 1) * num_1]
            else:
                index0 = indexs0[i][0][fold * num_0:]  # take the rest and put into the last fold
                index1 = indexs1[i][0][fold * num_1:]

            # the final samples in the cancer in that fold will have some of their driver gene mutated and some will
            # not(ensure every fold have mutated and not mutated driver gene in that cancer)
            final_index = index0 + index1
            sample = data.iloc[final_index]
            tmp_result.append(sample)
        result.append(pd.concat(tmp_result))

        print('The %d fold，The number of samples in this fold is %d' % (fold, len(result[-1])))

    if not os.path.exists('../../data/cross_valid'):
        os.makedirs('../../data/cross_valid')

    for i, item in enumerate(result):
        # test if we have accomplished the work here:
        print(i,np.sum(item[item["organ"] == "CESC"]["PIK3CA"] == 1))
        print(i, np.sum(item[item["organ"] == "CESC"]["PIK3CA"] == 0))
        # we save only the required gene column.sbs signature columns and organ column to the file
        save_item = item[cfg.SBS_NAMES + cfg.GENE_NAMES + ['organ']]
        static_data = dict(save_item['organ'].value_counts())

        # save the first k-1 folds to k-1 cross validation data file
        # we draw the class distribution of each fold here
        if i + 1 < k:
            plt_figure(static_data, 'cross_validation_%d' % i)
            save_item.to_csv(os.path.join('../../data/cross_valid', 'cross_validation_%d.csv' % i))
        else:
            # save the last 1 folds to validation data file
            plt_figure(static_data, 'validation_dataset')
            save_item.to_csv(os.path.join('../../data/cross_valid', 'validation_dataset.csv'))
    print('finished')


if __name__ == "__main__":
    # read the data
    train = pd.read_csv(cfg.DATA_PATH, usecols=cfg.SBS_NAMES + cfg.GENE_NAMES + ['organ'], low_memory=True)
    train = train.fillna(0)  # handling the NaN

    # we draw the cancer type distribution for all dataset here
    plt_figure(dict(train['organ'].value_counts()), 'all')

    # load the gene occurrence probability in each cancer
    cancer_prob = tool.obtain_gene_prob_cancer()
    # starting performing stratified sampling
    k_fold_split(train, cancer_prob,cfg.CROSS_VALIDATION_COUNT)
