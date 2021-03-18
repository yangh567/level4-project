"""

This file is used to split the original data into 6 folds
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


# read the data
train = pd.read_csv(cfg.DATA_PATH)
train = train.fillna(0)  # handling the NaN

# we draw the cancer type distribution for all dataset here
plt_figure(dict(train['organ'].value_counts()), 'all')


def k_fold_split(train_df, k):
    # start the stratified sampling
    types = list(train_df.groupby("organ"))  # we group the data using cancer types for future sampling
    result = []
    indexs = []

    # we generate the data from each of the cancer type and shuffle their index
    for cancer, data in types:
        index = list(range(len(data)))
        np.random.shuffle(index)
        indexs.append((index, int(len(data) / k)))

    for fold in range(k):
        tmp_result = []
        for i, type_ in enumerate(types):
            cancer, data = type_
            # find the index of the data we need to put into each fold
            num = indexs[i][1]

            if fold + 1 < k:
                index = indexs[i][0][fold * num: (fold + 1) * num]
            else:
                index = indexs[i][0][fold * num:]  # take the rest and put into the last fold
            sample = data.iloc[index]
            tmp_result.append(sample)
        result.append(pd.concat(tmp_result))
        print('The %d fold，The number of samples in this fold is %d' % (fold, len(result[-1])))

    if not os.path.exists(cfg.C_V_DATA_PATH):
        os.makedirs(cfg.C_V_DATA_PATH)

    for i, item in enumerate(result):

        # we save only the required gene column.sbs signature columns and organ column to the file
        save_item = item[cfg.SBS_NAMES + cfg.GENE_NAMES + ['organ']]
        static_data = dict(save_item['organ'].value_counts())

        # save the first k-1 folds to k-1 cross validation data file
        # we draw the class distribution of each fold here
        if i + 1 < k:
            plt_figure(static_data, 'cross_validation_%d' % i)
            save_item.to_csv(os.path.join(cfg.C_V_DATA_PATH, 'cross_validation_%d.csv' % i))
        else:
            # save the last 1 folds to validation data file
            plt_figure(static_data, 'validation_dataset')
            save_item.to_csv(os.path.join(cfg.C_V_DATA_PATH, 'validation_dataset.csv'))
    print('finished')


if __name__ == "__main__":
    k_fold_split(train, cfg.CROSS_VALIDATION_COUNT)
