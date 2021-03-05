"""
    using lasso regression train model on original dataset
    to study the HRDetect model (the data used is now discarded)(Part of research)
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Lasso

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve

import warnings
import sklearn.exceptions

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# *********config**********

# Donor Age,ER status,Gene,isBrcaMonoallelic,isKnownGermline,isNewGermline,IsSomaticMeth,
using_columns = ['SV%d' % i for i in range(1, 7)]

# Donor Age,ER status,Gene,isBrcaMonoallelic,isKnownGermline,isNewGermline,
# IsSomaticMeth,'ins', 'del.mh.prop', 'del.rep.prop',	'del.none.prop', 'hrd'
using_columns.extend(['ins', 'del.mh.prop', 'del.rep.prop', 'del.none.prop', 'hrd'])

# set the y here
label_column = 'Gene'

sub_columns = ['e.%d' % item for item in [1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26]]
using_columns.extend(sub_columns)

# 'SV5', 'SV6', 'SV2', 'e.8', 'e.2', 'e.17', 'e.1', 'e.18', 'e.13',
# 'hrd', 'e.20', 'ins', 'e.26', 'SV4', 'SV1', 'e.5', 'e.6', 'e.3', 'SV3'
normalize_columns = list(set(using_columns) - set(['del.mh.prop', 'del.rep.prop', 'del.none.prop']))

path_data = '../../data/raw/b_dataset.csv'
dataset = pd.read_csv(path_data)

# MARK The data contains the processed data of the original data. Let's try this unprocessed first here
# process the data

dataset[sub_columns] = np.log(dataset[sub_columns] + 1)

# standardization -- fitting in same scale
ss = StandardScaler()
dataset[normalize_columns] = ss.fit_transform(dataset[normalize_columns])

# lasso regression (560,22)
x = dataset[using_columns].fillna(0).to_numpy()

# Replace the label. Cases affected by BRCA1 / BRCA2 are 1, and those not affected are 0
y = dataset[label_column].map({'BRCA1': 1, 'BRCA2': 1}).fillna(0).to_numpy()

result = []
# using k-fold cross validation
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(x):
    train_x, train_y = x[train_index], y[train_index]
    test_x, test_y = x[test_index], y[test_index]
    lasso = Lasso(max_iter=10000, alpha=0.9, fit_intercept=True)  # Use the default parameters for now
    lasso.fit(train_x, train_y)
    pred_y = lasso.predict(test_x)

    pred_y[pred_y > 0.5] = 1
    pred_y[pred_y <= 0.5] = 0
    acc = accuracy_score(pred_y, test_y)
    cp = classification_report(test_y, pred_y)
    confusion_mat = confusion_matrix(test_y, pred_y)
    roc_c = roc_curve(test_y, pred_y)
    roc_acc_s = roc_auc_score(test_y, pred_y)
    # TODO store the data ，draw the graph
    print(acc)
    # print(lasso.coef_)
