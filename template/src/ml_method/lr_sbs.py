"""
    using logistic regression algorithm for classifying the cancer type
"""
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
# from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import argparse


def args():
    """
        # TODO
    :return:
    """


def get_clf(name='lr', cls_num=10):
    if name == 'lr':
        return LogisticRegression(max_iter=1000, multi_class='multinomial')
    elif name == 'svm':
        return SVC(kernel='rbf', C=0.4)     # Use radial basis functions


def start():
    path_data = '../../data/raw/TCGA_WES_sigProfiler_SBS_signatures_in_samples.csv'

    train = pd.read_csv(path_data)

    # extract data
    x = train[set(train.columns) - set(['Cancer Types', 'Sample Names'])]
    y = train['Cancer Types']

    # pre-process data
    # mm = MinMaxScaler()
    # x = mm.fit_transform(x)

    # build numerical label from type
    # le = MultiLabelBinarizer()
    le = LabelEncoder()
    y = le.fit_transform(y)
    # oh = OneHotEncoder()
    # y = oh.fit_transform(y.reshape(-1, 1))

    # TODO Stratified sampling
    # ss = StratifiedShuffleSplit(test_size=0.2, random_state=100)
    #
    # Split the training set into the test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=100)

    # clf = get_clf('svm')
    clf = get_clf()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_pred, y_test)
    confusion_mat = confusion_matrix(y_test, y_pred)
    print('Acc: %.4f' % acc)
    print(confusion_mat)
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    start()
