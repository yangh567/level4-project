

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
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import my_tools as tool
import my_config as cfg
import my_model


# read the data
o_data = pd.read_csv(cfg.DATA_PATH)
o_data = o_data.fillna(0)   # handling the NaN values
x = o_data[cfg.SBS_NAMES]
y = o_data[cfg.GENE_NAMES]

y[y >= 1] = 1
y[y <= 1] = 0
y = y.values


# data preprocessing

# # handling NaN values

# y = y.fillna(0).values


# x standardization

scaler = StandardScaler()
x = scaler.fit_transform(x)


# feature selection

# x = tool.feature_select(x, y)

# construct one-hot encoding for the gene mutation status
# we don't need it now as the gene mutation status is shown in one-hot encoding
# one_encoder = OneHotEncoder()
# y = one_encoder.fit_transform(y.reshape(y.shape[0], 1))

# constructing the training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

# # performing the classification

# model = RandomForestClassifier(n_estimators=10)
# # model = LogisticRegression(penalty='l2', C=1, multi_class='auto')
# model.fit(x_train, y_train)

# we setup the model as multi backpropagation network
model = my_model.MultiBPNet(x.shape[1], y.shape[1])

# we set the criteria as mean square error loss
criterion = nn.MSELoss()


# optimizer = torch.optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE)

# we set the optimizer as Adam
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)


# transform the dataframe into tensor
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
# x_train = torch.autograd.Variable(torch.tensor(x_train, dtype=torch.float32))
# x_test = torch.autograd.Variable(torch.tensor(x_test, dtype=torch.float32))
# y_train = torch.autograd.Variable(torch.tensor(y_train, dtype=torch.float32))
# y_test = torch.autograd.Variable(torch.tensor(y_test, dtype=torch.float32))




# for each epoch, the model will separate whole data into training and testing set and start training

    # for all batch in training set:
        # the input will be the batched samples
        # the target will be the cancer label of the batched samples
        # we calculate loss and optimize the model in every batch size until the training data is all used for training
        # the accuracy of each batch will be accumulated and taking average for the training accuracy

    # we evaluate the model by performing prediction on x_test
    # and compare it to the y_test to evaluate overall accuracy of the model
    # and we take greatest accuracy of testing set and corresponding trained parameter as perfectly trained model

save_data = [['epoch', 'loss', 'train accuracy', 'test accuracy', 'best test accuracy']]
best_acc = -1
batch_size = 32
batch_count = int(len(x_train) / batch_size) + 1
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

        y_pred = y_pred.detach().numpy()
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
        acc += accuracy_score(target.numpy(), y_pred)
        # acc += metrics.f1_score(target.numpy(), y_pred, average="macro")

    # print("Epoch: {}, Loss: {:.5f}, Accuracy: {:.5f}".format(epoch, epoch_loss / batch_count, acc / batch_count))

    # test
    model.eval()
    y_pred = model(x_test).detach().numpy()
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    acc_test = accuracy_score(y_test, y_pred)
    # f1_test = metrics.f1_score(y_test, y_pred, average="macro")
    if best_acc < acc_test:
        best_acc = acc_test
    print("Epoch: {}, Loss: {:.5f}, Train Accuracy: {:.5f}, Test Accuracy: {:.5f}, Best Test Accuracy: {:.5f}".format(
        epoch, epoch_loss / batch_count, acc / batch_count, acc_test, best_acc))
    save_data.append([epoch, epoch_loss / batch_count, acc / batch_count, acc_test, best_acc])

# evaluation
# y_hat = model.predict(x_test)
# print(metrics.f1_score(y_test, y_hat, average="macro"))
# print(metrics.f1_score(y_test, y_hat, average="weighted"))
# print('Accuracy score: %.6f' % accuracy_score(y_hat, y_test))

# save the weight of the sbs in each gene mutation and the bias of the genes calculated by model
weight = model.layer.weight.detach().numpy()
bias = model.layer.bias.detach().numpy()
if not os.path.exists('./result'):
    os.makedirs('./result')

df = pd.DataFrame(save_data)
df.to_csv("./result/gene-result.csv", index=False, header=False)

np.save("./result/gene-x.npy", x)
np.save("./result/gene-y.npy", y)
np.save("./result/gene_type-weight.npy", weight)
np.save("./result/gene_type-bias.npy", bias)
print('save weight file to ./result')
