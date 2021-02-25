import pandas as pd

import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import my_tools as tool
import my_config as cfg


def process_data(data, scale=True):
    encoder = LabelEncoder()

    x = data[cfg.SBS_NAMES]
    y = data['organ']
    if scale:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        y = encoder.fit_transform(y)
    return x, y


# concatenate data and handle data
def get_data(o_data, index):
    # concatenate data
    train = []
    test = None
    for i in range(len(o_data)):
        if i != index:
            train.append(o_data[i])
        else:
            test = o_data[i]

    # concatenate the 4 cross validation training dataframes
    # in the list to one cross validation training dataframe
    train = pd.concat(train)

    # assign (train_x,test_x) the sbs values and (train_y,test_y) the cancer type one hot encoding
    train_x, train_y = process_data(train)
    test_x, test_y = process_data(test)
    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    # read the data and append the 4 cross_validation data pandas frame into o_data list
    o_data = []
    for i in range(cfg.CROSS_VALIDATION_COUNT - 1):
        o_data.append(pd.read_csv(os.path.join(cfg.C_V_DATA_PATH, 'cross_validation_%d.csv' % i)))

    # read the validation data pandas frame
    valid_dataset = pd.read_csv(os.path.join(cfg.C_V_DATA_PATH, 'validation_dataset.csv'))

    # handling the NaN values in each of the data frame
    o_data = [item.fillna(0) for item in o_data]
    valid_dataset = valid_dataset.fillna(0)

    # make the cross validation testing and validation data
    test_acc = []
    valid_acc = []

    for i in range(cfg.CROSS_VALIDATION_COUNT - 1):

        model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=400, penalty='l2', C=1)

        # for each cross_validation dataset we have in o_data list
        # we select the ith validation dataset as testing data and remaining as training data
        train_x, train_y, test_x, test_y = get_data(o_data, i)

        # The last validation data to validate on model is obtained from validation dataset
        valid_x, valid_y = process_data(valid_dataset)

        # we then get the cross validation training and cross validation testing for training model (4:1),5 fold
        # we train the BPnet model based on those data set and
        # get average of training accuracy and (best testing accuracy which represent the best model accuracy)

        model.fit(train_x,train_y)

        pred_y = model.predict(test_x)
        acc = accuracy_score(pred_y, test_y)
        test_acc.append(acc)

        # then we evaluate the model on validation set
        # (the ratio between cross validation training set and validation set is 5:1,  5 fold cross validation)

        pred_valid_y = model.predict(valid_x)
        acc_valid = accuracy_score(pred_valid_y, valid_y)
        valid_acc.append(acc_valid)

        print('The %d fold，The best trained model has testing accuracy of  %.4f，The validation accuracy for this fold '
              'is '
              '%.4f' % (i, test_acc[-1], valid_acc[-1]))