"""
    sbs
    rs
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import eli5
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


model = 'rf'    # The model that we use (logistic regression)
k = 10  # This is the time we do cross validation
path_data = '../../data/processed/sample_id.sbs.organ.csv'


def get_model(name='lr', lr=1e-3):
    if name == 'lr':
        return LogisticRegression(multi_class='multinomial', max_iter=10000)
    elif name == 'rf':
        return RandomForestClassifier()
    elif name == 'svm':
        return SVC()
    elif name == 'lasso':
        # return LassoCV(eps=lr, max_iter=10000)
        return LassoCV(eps=lr, max_iter=10000, cv=10)


train = pd.read_csv(path_data)

train = train.fillna(0)

# construct the label
x = train[set(train.columns) - set(['Sample_ID', 'organ'])]
feature_set = set(train.columns) - set(['Sample_ID', 'organ'])

# this is the number of sbs signatures
print(len(feature_set))

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

clf = get_model(model)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100, shuffle=True)



clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
y_pred = np.around(y_pred).astype('int')

report = classification_report(y_test, y_pred)

print('The classification report')
print(report)

if model == 'lr' or model == 'lasso':
    print('The classification intercept ：')
    print(clf.intercept_)
    print('The classification coefficient aka pattern ：')
    print(clf.coef_)

elif model == 'rf':
    print('pattern')
    # mean decrease impurity
    for item in sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_),
                           list(set(train.columns) - set(['Sample_ID', 'organ']))), reverse=True):
        print(item)
    print(feature_set)
    #print(eli5.format_as_text(eli5.explain_weights(clf,feature_names=feature_set,target_names = train['organ'] )))


