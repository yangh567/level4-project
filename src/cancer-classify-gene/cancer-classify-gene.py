"""

    This file is used to test on the self-build model on the classification_cancer_analysis of genes
    based on mutation signature (SBS) using 5 fold cross validation (DEPRECATED RESEARCH)

"""
# load the module
import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

# concatenate the path of utilities
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('..', 'my_utilities')))
# load the model, configuration and tools here
from my_utilities import my_config as cfg
from my_utilities import my_model as my_model
from my_utilities import my_tools as tool

# ignore the warnings
import warnings

warnings.filterwarnings('ignore')


# the function to process the data to separate the label and features as well as doing data transformation
def process_data(data):
    encoder = LabelEncoder()
    x = data['organ']
    y = data[cfg.GENE_NAMES]
    y[y >= 1] = 1
    y[y < 1] = 0
    x = encoder.fit_transform(x).reshape(-1, 1)
    y = y.values
    return x, y


# the function to obtain the training_x,testing_x,training_y and testing_y
def get_data(all_data, index):
    train_data = []
    test = None
    for i in range(len(all_data)):
        if i != index:
            train_data.append(all_data[i])
        else:
            test = all_data[i]
    train_data = pd.concat(train_data)
    train_data_x, train_data_y = process_data(train_data)
    test_data_x, test_data_y = process_data(test)
    return train_data_x, train_data_y, test_data_x, test_data_y


# the function used to train on the model and test on the model in each fold
def train(train_x, train_y, test_x, test_y, fold):
    x_train = torch.tensor(train_x, dtype=torch.float32)
    y_train = torch.tensor(train_y, dtype=torch.float32)
    batch_size = cfg.BATCH_SIZE
    batch_count = int(len(x_train) / batch_size) + 1

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

            # doing gradient descent here
            y_pred = model(inputs)
            loss = criterion(y_pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            y_pred = y_pred.detach().numpy()
            y_pred[y_pred >= 0.5] = 1
            y_pred[y_pred < 0.5] = 0

            # accumulate the training accuracy in eac batch and take average to reflect on the model
            acc += np.mean(np.sum((target.detach().numpy() - y_pred) == 0, axis=0) / target.detach().numpy().shape[0])

        # test the accuracy here
        print("Epoch: {}, Loss: {:.5f}, Train Accuracy: {:.5f}".
              format(epoch, epoch_loss / batch_count, acc / batch_count))
        save_data.append([epoch, epoch_loss / batch_count, acc / batch_count])
    acc_test = score(test_x, test_y)
    print("The cross-validation test accuracy on fold " + str(fold) + " is :", acc_test)
    return acc_test


# the function to score the classification accuracy as well as collecting the classification score of each individual
# gene mutation status
def score(test_x, test_y, title=0, gene_list=None, gene_list_mutation_prob=None, final=False):
    model.eval()
    # as the gene is not separated among classes so when give empty string to form the png file
    cancer_type = ""
    x_test = torch.tensor(test_x, dtype=torch.float32)
    y_test = torch.tensor(test_y, dtype=torch.float32)

    y_pred = model(x_test)

    y_pred = y_pred.detach().numpy()
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0

    acc_test = np.mean(np.sum((y_test.detach().numpy() - y_pred) == 0, axis=0) / y_test.detach().numpy().shape[0])

    # only for validation data
    if final:
        # collecting the classification score of each individual gene mutation status
        tool.gene_class_report(y_test.detach().numpy(), y_pred, cancer_type, title, gene_list, gene_list_mutation_prob)
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

    # set up the 5 fold cross validation
    for i in range(cfg.CROSS_VALIDATION_COUNT - 1):
        train_x, train_y, test_x, test_y = get_data(o_data, i)
        valid_x, valid_y = process_data(valid_dataset)

        # set up model,loss function,optimizers
        model = my_model.MultiBPNet(train_x.shape[1], train_y.shape[1])
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

        test_acc.append(train(train_x, train_y, test_x, test_y, i))
        valid_acc.append(
            score(valid_x, valid_y, title=i, gene_list=cfg.GENE_NAMES, gene_list_mutation_prob=None, final=True))
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

    # display the results
    print('The 5 fold cross validation has 5 testing result,they are :', test_acc)
    print('The validation accuracies for 5 fold cross validation are :', valid_acc)
