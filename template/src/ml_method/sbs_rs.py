"""
    sbs
    rs
"""
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LassoCV
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from sklearn.model_selection import cross_val_score
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


model = 'lr'    # The model that we use (logistic regression)
k = 10  # This is the time we do cross validation
path_data = '../../data/raw/cancer_signature_data.csv'


def get_model(name='lr', lr=1e-3):
    if name == 'lr':
        return LogisticRegression(multi_class='multinomial', max_iter=10000)
    elif name == 'rf':
        return RandomForestClassifier()
    elif name == 'svm':
        return SVC()
    elif name == 'lasso':
        # return LassoCV(eps=lr, max_iter=10000)
        return MultiTaskLassoCV(eps=lr, max_iter=10000)


train = pd.read_csv(path_data)

# construct the label
x = train[set(train.columns) - set(['id', 'organ'])]
feature_set = set(train.columns) - set(['id', 'organ'])

x = x.values   # find data
# standardization
# z = (x - u) / s
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Separate the labels from the id.
# Those less than or equal to 10 have tumors, and 11-29 have no tumor

# y = pd.get_dummies(train['organ'])
le = LabelEncoder()
y = le.fit_transform(train['organ'])

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100, shuffle=True)


# roc_c = roc_curve(y_test, y_pred)
# roc_acc_s = roc_auc_score(y_test, y_pred)


# we perform the cross validation here
clf = get_model(model)
score = cross_val_score(clf, x, y, cv=10, n_jobs=8, scoring='accuracy')
# clf.fit(x, y)
print("These are the scores for each fold :", score)
print("This is the average score for all fold :", score.mean())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100, shuffle=True)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
score = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
# This is the accuracy after cross validation
print('final test: %.6f' % score)
print('The classification report')
print(report)
print('The classification intercept ：')
print(clf.intercept_)
print('The classification coefficient aka the patterns ：')


# now we check which feature do have max weight
import operator

for i in range(21):
    coef_dict = {}
    for coef, feat in zip(clf.coef_[i,:],feature_set):
        coef_dict[feat] = coef
    print(max(coef_dict.items(), key=operator.itemgetter(1))[0])
# result = {}
# clf = get_model(model)
# kf = KFold(n_splits=k)
# for train_index, test_index in kf.split(x):
#     x_train, y_train = x[train_index], y[train_index]
#     x_test, y_test = x[test_index], y[test_index]
#     # clf = get_model(model)
#     clf.fit(x_train, y_train)
#     y_pred = clf.predict(x_test)
#     acc = accuracy_score(y_pred, y_test)
#     cm = confusion_matrix(y_test, y_pred)
#     if 'acc' in result.keys():
#         result['acc'] += acc
#         # result['cm'] += cm
#     else:
#         result['acc'] = acc
#         # result['cm'] = cm
#     print(acc)

# print('model name: ', model)
# print('Acc: %.4f' % (result['acc'] / k))
# print(cp)
# print(result['cm'] / k)
# print(roc_c)
# print(roc_acc_s)

