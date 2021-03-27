"""

    This file is used to test on the self-build model on classification_cancer_analysis of BLCA and BRCA
    based on mutation signature (SBS) using 5 fold cross validation to ensure the possibility of classifying 32 cancers

"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('..','my_utilities')))

from my_utilities import my_config as cfg
from my_utilities import my_model as my_model
from my_utilities import my_confusion_matrix as m_c_m


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

figure_data = './result/cancer_classification_confusion_matrix'

if not os.path.exists(figure_data):
    os.makedirs(figure_data)


# function used to get the x and y and scale them
def process_data(data, scale=True):
    x = data[(data['organ'] == "BLCA") | (data['organ'] == "LGG")][cfg.SBS_NAMES]
    y = data[(data['organ'] == "BLCA") | (data['organ'] == "LGG")]["organ"]
    if scale:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        y = pd.get_dummies(y).values
    return x, y


# concatenate the data here and ensure each time the test fold and training fold is different
def get_data(o_data, index):
    # concatenate the data
    train = []
    test = None
    for i in range(len(o_data)):
        if i != index:
            train.append(o_data[i])
        else:
            test = o_data[i]
    train = pd.concat(train)
    train_x, train_y = process_data(train)
    test_x, test_y = process_data(test)
    return train_x, train_y, test_x, test_y


# for each epoch, the model will separate whole data into training and testing set and start training

# for all batch in training set:
# the input will be the batched samples
# the target will be the cancer label of the batched samples
# we calculate loss and optimize the model in every batch size until the training data is all used for training
# the accuracy of each batch will be accumulated and taking average for the training accuracy

# we evaluate the model by performing prediction on x_test
# and compare it to the y_test to evaluate overall accuracy of the model
# and we take greatest accuracy of testing set and corresponding trained parameter as perfectly trained model


# we start training the model here using batch Adam optimizer BPnet
def train(train_x, train_y, test_x, test_y):
    x_train = torch.tensor(train_x, dtype=torch.float32)
    y_train = torch.tensor(train_y, dtype=torch.float32)

    # The training set will be separated into mini batches for improving calculation
    batch_size = 16
    batch_count = int(len(x_train) / batch_size) + 1

    best_acc = -1.
    save_data = [['epoch', 'loss', 'train accuracy', 'test accuracy', 'best test accuracy']]
    for epoch in range(cfg.CANCER_EPOCH):
        model.train()
        epoch_loss = 0
        acc = 0
        for i in range(batch_count):
            if i * batch_size >= len(x_train):
                continue
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
        acc_test = score(test_x, test_y)
        if best_acc < acc_test:
            best_acc = acc_test
        print("Epoch: {}, Loss: {:.5f}, Train Accuracy: {:.5f}, Test Accuracy: {:.5f}, Best Test Accuracy: {:.5f}".
              format(epoch, epoch_loss / batch_count, acc / batch_count, acc_test, best_acc))
        save_data.append([epoch, epoch_loss / batch_count, acc / batch_count, acc_test, best_acc])
    return best_acc


def score(test_x, test_y, title=0, report=False):
    model.eval()
    x_test = torch.tensor(test_x, dtype=torch.float32)
    y_test = torch.tensor(test_y, dtype=torch.float32)

    y_pred = model(x_test)
    y_pred = torch.argmax(y_pred, dim=1).detach().numpy()

    acc_test = accuracy_score(torch.argmax(y_test, dim=1), y_pred)
    if report:
        m_c_m.plot_confusion_matrix(torch.argmax(y_test, dim=1), y_pred, title)
        print(classification_report(torch.argmax(y_test, dim=1), y_pred))
    return acc_test


if __name__ == '__main__':
    # read the data here
    o_data = []
    for i in range(cfg.CROSS_VALIDATION_COUNT - 1):
        o_data.append(pd.read_csv(os.path.join(cfg.C_V_DATA_PATH, 'cross_validation_%d.csv' % i)))
    valid_dataset = pd.read_csv(os.path.join(cfg.C_V_DATA_PATH, 'validation_dataset.csv'))

    # handling the NaN values
    o_data = [item.fillna(0) for item in o_data]
    valid_dataset = valid_dataset.fillna(0)

    # make the 5-fold cross validation here
    test_acc = []
    valid_acc = []

    for i in range(cfg.CROSS_VALIDATION_COUNT - 1):
        # sbs num : 49,cancer types num : 2
        model = my_model.SoftMaxBPNet(49, 2)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

        train_x, train_y, test_x, test_y = get_data(o_data, i)
        valid_x, valid_y = process_data(valid_dataset)

        test_acc.append(train(train_x, train_y, test_x, test_y))
        valid_acc.append(score(valid_x, valid_y, title=i, report=True))
        print('The %d fold，The best testing accuracy for trained model at this fold is %.4f，The validation accuracy '
              'for this fold is %.4f' % (i, test_acc[-1], valid_acc[-1]))

        # we save the weight of each sbs signatures for each cancer type
        weight = model.layer.weight.detach().numpy()
        bias = model.layer.bias.detach().numpy()
        if not os.path.exists('./result'):
            os.makedirs('./result')
        np.save("./result/cancer_type-weight_%d.npy" % i, weight)
        np.save("./result/cancer_type-bias_%d.npy" % i, bias)
        print('save weight file to ./result')

    print('The 5 fold cross validation has 5 testing result,they are :', test_acc)
    print('The validation accuracies for 5 fold cross validation are :', valid_acc)
