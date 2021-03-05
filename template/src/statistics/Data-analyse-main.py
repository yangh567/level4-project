# -*- encoding: utf-8 -*-

"""
    this is the code used for statics analysis(DEPRECATED)(Part of research)
"""

import pandas as pd

path_data = '../../data/raw/GDC-PANCAN.muse_snv.tsv'
# path_data = '../dataset/example/SomaticMutation/GDC-PANCAN.muse_snv_sample.tsv'
train = pd.read_csv(path_data, sep='\t')

print('dataset shape: ', train.shape)
for column in train.columns:
    column_set = list(set(train[column].values))
    # count all of the individual instances in all features
    print('\ncolumn %s, type count: %d' % (column, len(column_set)))
    if len(column_set) < 10:
        print(' '.join(column_set))

labels = []
for item in train['Sample_ID']:
    labels.append(item.strip().split('-')[-1])
labels = list(set(labels))
# count all of the individual tumor labels
print('label count: ', len(labels))
for label in labels:
    print(label)
