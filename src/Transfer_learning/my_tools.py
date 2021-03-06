from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import my_config as cfg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def feature_select(x, y):
    """

    :return:
    """
    clf = ExtraTreesClassifier()
    clf.fit(x, y)
    # print(clf.feature_importances_)
    model = SelectFromModel(clf, prefit=True)
    x_new = model.transform(x)
    return x_new


class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode

    def fit(self, X, y=None):
        return self  # not relevant here

    def transform(self, X):
        """
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        """
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# used to print out the classification result for each genes
def gene_class_report(y, y_hat, title):
    gene_accuracy_dict = {}
    gene_list = cfg.GENE_NAMES[:32]
    gene_accuracy_list = list(np.sum((y - y_hat) == 0, axis=0) / y.shape[0])

    for i in range(len(gene_list)):
        gene_accuracy_dict[gene_list[i]] = gene_accuracy_list[i]

    df = pd.DataFrame(list(gene_accuracy_dict.items()), columns=['gene_name', 'accuracy'])
    if not os.path.exists('./result/gene_classification_accuracy'):
        os.makedirs('./result/gene_classification_accuracy')

    df.to_csv('./result/gene_classification_accuracy/The_classification_across_gene_fold_%d.csv' % title)
    return gene_accuracy_dict
    pass
