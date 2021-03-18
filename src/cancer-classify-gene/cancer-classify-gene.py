"""

    This file is used to test on the self-build model on the classification_cancer_gene_analysis of genes
    based on mutation signature (SBS) using 5 fold cross validation

"""
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import seaborn as sns

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('..','my_utilities')))


#import my_utilities as my_u
from my_utilities import my_config as cfg
from my_utilities import my_model as my_model
from my_utilities import my_tools as tool
#from my_utilities import my_tools as my_model
#from  my_utilities import my_tools as tool
#from my_utilities import my_model as my_model
#from my_utilities import my_tools as tool
# from src.my_utilities import my_config as cfg
# from src.my_utilities import my_model as my_model
# from src.my_utilities import my_tools as tool
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')


def process_data(data, scale=True):
    encoder = LabelEncoder()
    x = data['organ']
    y = data[cfg.GENE_NAMES]
    y[y >= 1] = 1
    y[y < 1] = 0
    x = encoder.fit_transform(x).reshape(-1, 1)
    y = y.values
    return x, y


def get_data(o_data, index):
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


def train(train_x, train_y, test_x, test_y):
    x_train = torch.tensor(train_x, dtype=torch.float32)
    y_train = torch.tensor(train_y, dtype=torch.float32)
    batch_size = cfg.BATCH_SIZE
    batch_count = int(len(x_train) / batch_size) + 1

    best_acc = -1.
    save_data = [['epoch', 'loss', 'train accuracy', 'test accuracy', 'best test accuracy']]
    for epoch in range(cfg.EPOCH):
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

            y_pred = y_pred.detach().numpy()
            # y_pred = torch.argmax(y_pred, dim=1).detach().numpy()
            y_pred[y_pred >= 0.5] = 1
            y_pred[y_pred < 0.5] = 0

            acc += np.mean(np.sum((target.detach().numpy() - y_pred) == 0, axis=0) / target.detach().numpy().shape[0])
            # acc += accuracy_score(torch.argmax(target, dim=1), y_pred)

        acc_test = score(test_x, test_y)
        if best_acc < acc_test:
            best_acc = acc_test
        print("Epoch: {}, Loss: {:.5f}, Train Accuracy: {:.5f}, Test Accuracy: {:.5f}, Best Test Accuracy: {:.5f}".
              format(epoch, epoch_loss / batch_count, acc / batch_count, acc_test, best_acc))
        save_data.append([epoch, epoch_loss / batch_count, acc / batch_count, acc_test, best_acc])
    return best_acc


def score(test_x, test_y, title=0,gene_list=None,gene_list_mutation_prob=None, final=False):
    model.eval()
    cancer_type=""
    x_test = torch.tensor(test_x, dtype=torch.float32)
    y_test = torch.tensor(test_y, dtype=torch.float32)

    y_pred = model(x_test)
    # y_pred = torch.argmax(y_pred, dim=1).detach().numpy()

    y_pred = y_pred.detach().numpy()
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0

    acc_test = np.mean(np.sum((y_test.detach().numpy() - y_pred) == 0, axis=0) / y_test.detach().numpy().shape[0])
    # print(y_pred.shape[1])
    if final:
        tool.gene_class_report(y_test.detach().numpy(), y_pred, cancer_type,title,gene_list,gene_list_mutation_prob)
    return acc_test


if __name__ == '__main__':
    # read the data
    o_data = []
    for i in range(cfg.CROSS_VALIDATION_COUNT - 1):
        o_data.append(pd.read_csv(os.path.join(cfg.C_V_DATA_PATH, 'cross_validation_%d.csv' % i)))
    valid_dataset = pd.read_csv(os.path.join(cfg.C_V_DATA_PATH, 'validation_dataset.csv'))

    # handling the NaN
    o_data = [item.fillna(0) for item in o_data]
    valid_dataset = valid_dataset.fillna(0)

    # set the recorder to record the trained model's best testing accuracy in each fold
    test_acc = []
    valid_acc = []

    for i in range(cfg.CROSS_VALIDATION_COUNT - 1):
        train_x, train_y, test_x, test_y = get_data(o_data, i)
        valid_x, valid_y = process_data(valid_dataset)

        model = my_model.MultiBPNet(train_x.shape[1], train_y.shape[1])
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

        test_acc.append(train(train_x, train_y, test_x, test_y))
        valid_acc.append(score(valid_x, valid_y, title=i, gene_list=cfg.GENE_NAMES,gene_list_mutation_prob=None,final=True))
        print('The %d fold，The best testing accuracy for trained model at this fold is %.4f，the validation accuracy '
              'for this fold is %.4f' % (i, test_acc[-1], valid_acc[-1]))

        # save the weight of each sbs
        weight = model.layer.weight.detach().numpy()
        bias = model.layer.bias.detach().numpy()
        if not os.path.exists('./result'):
            os.makedirs('./result')

        np.save("./result/gene_type-weight_%d.npy" % i, weight)
        np.save("./result/gene_type-bias_%d.npy" % i, bias)
        print('save weight file to ./result')

    print('The 5 fold cross validation has 5 testing result,they are :', test_acc)
    print('The validation accuracies for 5 fold cross validation are :', valid_acc)
