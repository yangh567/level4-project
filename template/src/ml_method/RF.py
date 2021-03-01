# -*- encoding: utf-8 -*-
"""
the classification based on random forest classification
(The data is used is now discarded)(Part of research)
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#path_data = '../../data/raw/GDC-PANCAN.muse_snv.tsv'

path_data = '../../data/raw/GDC-PANCAN.muse_snv_short.tsv'
train = pd.read_csv(path_data, sep='\t')

# construct the label
x = pd.get_dummies(train[set(train.columns) - set(['Sample_ID', 'dna_vaf'])]).join(train['dna_vaf'])    # construct the one-hot-coding
x = x.values   # find the data

# we seperate the cancer label(whether it is cancer) from tumor id,
# Those less than or equal to 10 have tumors, and 11-29 have no tumors.
y = list(map(lambda x: 1 if int(x.strip()[13:15]) <= 10 else 0, train['Sample_ID']))

# seperate the training data into training and testing data for validation

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=100)

clf = RandomForestClassifier()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print(y_pred)
acc = accuracy_score(y_pred, y_test)
confusion_mat = confusion_matrix(y_test, y_pred)
print('Acc: %.4f' % acc)
print(confusion_mat)
print(classification_report(y_test, y_pred))
