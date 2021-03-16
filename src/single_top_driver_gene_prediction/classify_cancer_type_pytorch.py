"""

    This file is used to test on the self-build model on the logistic regression classification of cancers
    based on mutation signature (SBS) without using 5 fold cross validation

"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from my_confusion_matrix import plot_confusion_matrix
from sklearn.metrics import roc_curve, auc

import my_tools as tool
import my_config as cfg
import my_model
import warnings

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

figure_data = './result/cancer_classification_confusion_matrix'

if not os.path.exists(figure_data):
    os.makedirs(figure_data)


# function used to get the x and y and scale them
def process_data(data, scale=True):
    x = data[cfg.SBS_NAMES]
    y = data['organ']
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
def train_and_test(train_x, train_y, test_x, test_y,fold):
    x_train = torch.tensor(train_x, dtype=torch.float32)
    y_train = torch.tensor(train_y, dtype=torch.float32)

    # The training set will be separated into mini batches for improving calculation
    batch_size = 12
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

            y_pred = torch.argmax(y_pred, dim=1).detach().numpy()
            acc += accuracy_score(torch.argmax(target, dim=1), y_pred)
        print("Epoch: {}, Loss: {:.5f}, Train Accuracy: {:.5f}".
              format(epoch, epoch_loss / batch_count, acc / batch_count))
        save_data.append([epoch, epoch_loss / batch_count, acc / batch_count])
    acc_test = score(test_x, test_y)
    print("The cross-validation test accuracy on fold "+str(fold)+" is :",acc_test)

    return acc_test


def score(test_x, test_y, title=0, report=False):
    model.eval()
    x_test = torch.tensor(test_x, dtype=torch.float32)
    y_test = torch.tensor(test_y, dtype=torch.float32)

    y_pred = model(x_test)

    y_p = y_pred.detach().numpy()
    y_t = y_test.detach().numpy()

    y_pred = torch.argmax(y_pred, dim=1).detach().numpy()

    acc_test = accuracy_score(torch.argmax(y_test, dim=1).detach().numpy(), y_pred)
    if report:
        n_classes = len(y_test[1])
        plot_confusion_matrix(torch.argmax(y_test, dim=1).detach().numpy(), y_pred, title)
        print(classification_report(torch.argmax(y_test, dim=1).detach().numpy(), y_pred))

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_t[:, i], y_p[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot of a ROC curve for a specific class
        lg = 0
        plt.figure()
        for j in range(n_classes):
            plt.plot(fpr[j], tpr[j], label='ROC curve ' + cfg.ORGAN_NAMES[j] + ' (area = %0.2f)' % roc_auc[j])
            lg = plt.legend(bbox_to_anchor=(1.0, 1.0), loc='best', prop={'size': 6})
            plt.tight_layout()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        if not os.path.exists('./result/cancer_classification_roc_auc'):
            os.makedirs('./result/cancer_classification_roc_auc')
        plt.savefig(
            './result/cancer_classification_roc_auc/The_roc_auc_for_validation_in_fold_%d.png' % title, dpi=300,
            format='png',
            bbox_extra_artists=(lg,),
            bbox_inches='tight')

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
        # sbs num : 49,cancer types num : 32
        model = my_model.BPNet(49, 32)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

        train_x, train_y, test_x, test_y = get_data(o_data, i)
        valid_x, valid_y = process_data(valid_dataset)

        test_acc.append(train_and_test(train_x, train_y, test_x, test_y,i))
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
