"""

    This file is used to test on the self-build model on the classification of genes
    based on mutation signature (SBS) using 5 fold cross validation

"""

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import seaborn as sns

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import my_tools as tool
import my_config as cfg
import my_model

import warnings

warnings.filterwarnings('ignore')


def process_data(data, cancer_type, gene_list, scale=True):
    x = data[data["organ"] == cancer_type][cfg.SBS_NAMES]
    y = data[data["organ"] == cancer_type][gene_list]
    y[y >= 1] = 1
    y[y < 1] = 0
    y = y.values

    if scale:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
    return x, y


def get_data(o_data, index, cancer_type, gene_list):
    train = []
    test = None
    for i in range(len(o_data)):
        if i != index:
            train.append(o_data[i])
        else:
            test = o_data[i]
    train = pd.concat(train)
    train_x, train_y = process_data(train, cancer_type, gene_list)
    test_x, test_y = process_data(test, cancer_type, gene_list)
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


def train(train_x, train_y, test_x, test_y,fold):
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
            y_pred[y_pred > 0.5] = 1
            y_pred[y_pred <= 0.5] = 0

            acc += np.mean(np.sum((target.detach().numpy() - y_pred) == 0, axis=0) / target.detach().numpy().shape[0])
            # acc += accuracy_score(torch.argmax(target, dim=1), y_pred)

        print("Epoch: {}, Loss: {:.5f}, Train Accuracy: {:.5f}".
              format(epoch, epoch_loss / batch_count, acc / batch_count))
        save_data.append([epoch, epoch_loss / batch_count, acc / batch_count])
    acc_test = score(test_x, test_y)
    print("The cross-validation test accuracy on fold " + str(fold) + " is :", acc_test)
    return acc_test


def score(test_x, test_y, title=0, cancer__type="", gene_list=None,gene_list_mutation_prob=None,final=False):
    model.eval()
    x_test = torch.tensor(test_x, dtype=torch.float32)
    y_test = torch.tensor(test_y, dtype=torch.float32)

    y_pred = model(x_test)
    # y_pred = torch.argmax(y_pred, dim=1).detach().numpy()

    y_pred = y_pred.detach().numpy()
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0

    acc_test = np.mean(np.sum((y_test.detach().numpy() - y_pred) == 0, axis=0) / y_test.detach().numpy().shape[0])
    if final:
        tool.gene_class_report(y_test.detach().numpy(), y_pred, cancer__type, title,gene_list,gene_list_mutation_prob)
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
    test_acc_fold = []
    valid_acc = []
    valid_acc_fold = []

    gene_prob = pd.read_csv('./result/gene_prob.csv')
    cancer_prob = {}
    for name, item in gene_prob.groupby('cancer type'):
        cancer_prob[name] = item

    for fold in range(cfg.CROSS_VALIDATION_COUNT - 1):

        for cancer_type in cfg.ORGAN_NAMES:
            # gene_list = []
            # gene_list_mutation_prob = []
            gene_list_for_cancer = []
            gene_freq_list_for_cancer = []

            gene_list_final_for_cancer = []
            gene_freq_list_final_for_cancer = []

            for gene in cfg.GENE_NAMES:
                gene_list_for_cancer.append((gene, cancer_prob[cancer_type][gene].values[0]))
                gene_freq_list_for_cancer.append(cancer_prob[cancer_type][gene].values[0])

            # find the top 10 gene's index in pandas frame
            top_10_index = list(reversed(
                sorted(range(len(gene_freq_list_for_cancer)), key=lambda i: gene_freq_list_for_cancer[i])[-10:]))

            # find those gene and their freq as (gene,freq)
            res_list = [gene_list_for_cancer[i] for i in top_10_index]

            # append the gene name into gene_list_final_for_cancer list
            # append the gene mutation frequency to gene_freq_list_final_for_cancer list
            for (a, b) in res_list:
                gene_list_final_for_cancer.append(a)
                gene_freq_list_final_for_cancer.append(b)

            train_x, train_y, test_x, test_y = get_data(o_data, fold, cancer_type, gene_list_final_for_cancer)

            valid_x, valid_y = process_data(valid_dataset, cancer_type, gene_list_final_for_cancer)

            model = my_model.MultiBPNet(train_x.shape[1], train_y.shape[1])

            criterion = nn.MultiLabelSoftMarginLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

            test_acc.append(train(train_x, train_y, test_x, test_y,fold))
            valid_acc.append(score(valid_x, valid_y, title=fold, cancer__type=cancer_type, gene_list=gene_list_final_for_cancer,
                                   gene_list_mutation_prob=gene_freq_list_final_for_cancer,final=True))
            print('The %d fold，The best testing accuracy for trained model for %s at this fold is %.4f，the validation '
                  'accuracy '
                  'for this fold is %.4f' % (fold, cancer_type, test_acc[-1], valid_acc[-1]))

            # save the weight of each sbs for each highly frequented gene in that cancer
            weight = model.layer.weight.detach().numpy()
            bias = model.layer.bias.detach().numpy()
            if not os.path.exists('./result'):
                os.makedirs('./result')

            np.save("./result/gene_sbs_weights/gene_type-weight_in_fold%d_for_%s.npy" % (fold,cancer_type), weight)
            np.save("./result/gene_sbs_weights/gene_type-bias_in_fold%d_for_%s.npy" % (fold,cancer_type), bias)
            print('save weight file to ./result')
        test_acc_fold.append(np.mean(test_acc))
        valid_acc_fold.append(np.mean(valid_acc))
    print('The 5 fold cross validation has 5 testing across all 32 cancers result,they are :', test_acc_fold)
    print('The validation accuracies for 5 fold cross validation across all 32 cancers result,they are :', valid_acc_fold)
