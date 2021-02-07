import pandas as pd

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
import my_tools as tool
import my_config as cfg


# reading the data
o_data = pd.read_csv(cfg.DATA_PATH)
x = o_data[cfg.SBS_NAMES]
y = o_data[cfg.CANCER_TYPES_NAMES]

# Data preprocessing


# x standardization
scaler = StandardScaler()
x = scaler.fit_transform(x)

# encoding y into numbers
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# feature selection
# x = tool.feature_select(x, y)

# construct one-hot
# one_encoder = OneHotEncoder()
# y = one_encoder.fit_transform(y.reshape(y.shape[0], 1))

# construct the training and testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

# classification
# model = RandomForestClassifier(n_estimators=10)
model = LogisticRegression(penalty='l2', C=1)
model.fit(x_train, y_train)

# evaluation
y_hat = model.predict(x_test)
print('Accuracy score: %.6f' % accuracy_score(y_hat, y_test))

# TODO
# the variables to be kept
coef = model.coef_
intercept = model.intercept_
model.classes_ = model.classes_
