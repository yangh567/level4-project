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
from sklearn.metrics import accuracy_score
from sklearn import metrics

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import my_tools as tool
import my_config as cfg
import my_model


# reading the data
o_data = pd.read_csv(cfg.DATA_PATH)
o_data = o_data.fillna(0)   # handling NA
x = o_data[cfg.SBS_NAMES]
# y = o_data[cfg.CANCER_TYPES_NAMES]
print(o_data.columns.values.tolist())
y = o_data['organ']

# data preprocessing
# x standardization
scaler = StandardScaler()
x = scaler.fit_transform(x)


# feature selection
# x = tool.feature_select(x, y)
# encoding y into numbers
y = pd.get_dummies(y).values

# construct the training and testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

# # classification
# model = RandomForestClassifier(n_estimators=10)
# # model = LogisticRegression(penalty='l2', C=1, multi_class='auto')
# model.fit(x_train, y_train)
model = my_model.BPNet(x.shape[1], y.shape[1])
criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
# x_train = torch.autograd.Variable(torch.tensor(x_train, dtype=torch.float32))
# x_test = torch.autograd.Variable(torch.tensor(x_test, dtype=torch.float32))
# y_train = torch.autograd.Variable(torch.tensor(y_train, dtype=torch.float32))
# y_test = torch.autograd.Variable(torch.tensor(y_test, dtype=torch.float32))

batch_size = 16
batch_count = int(len(x_train) / batch_size) + 1

best_acc = -1.
for epoch in range(cfg.EPOCH):
    # train
    model.train()
    epoch_loss = 0
    acc = 0
    for i in range(batch_count):
        inputs = torch.autograd.Variable(x_train[i * batch_size: (i + 1) * batch_size])
        target = torch.autograd.Variable(y_train[i * batch_size: (i + 1) * batch_size])
        y_pred = model(inputs)
        loss = criterion(y_pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        y_pred = torch.argmax(y_pred, dim=1).detach().numpy()
        acc += accuracy_score(torch.argmax(target, dim=1), y_pred)
        # acc += metrics.f1_score(target.numpy(), y_pred, average="macro")

    # print("Epoch: {}, Loss: {:.5f}, Accuracy: {:.5f}".format(epoch, epoch_loss / batch_count, acc / batch_count))

    # test
    model.eval()
    y_pred = model(x_test)
    y_pred = torch.argmax(y_pred, dim=1).detach().numpy()
    # y_pred[y_pred > 0.5] = 1
    # y_pred[y_pred <= 0.5] = 0
    acc_test = accuracy_score(torch.argmax(y_test, dim=1), y_pred)
    # f1_test = metrics.f1_score(y_test, y_pred, average="macro")
    if best_acc < acc_test:
        best_acc = acc_test
    print("Epoch: {}, Loss: {:.5f}, Train Accuracy: {:.5f}, Test Accuracy: {:.5f}, Best Accuracy: {:.5f}".format(
        epoch, epoch_loss / batch_count, acc / batch_count, acc_test, best_acc))

# keep the models
weight = model.layer.weight.detach().numpy()
bias = model.layer.bias.detach().numpy()
if not os.path.exists('./result'):
    os.makedirs('./result')
np.save("./result/cancer_type-weight.npy", weight)
np.save("./result/cancer_type-bias.npy", bias)
print('save weight file to ./result')
