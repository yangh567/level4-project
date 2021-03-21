"""

    This file is used to test on the self-build model on the classification_cancer_gene_analysis of top 5 frequently mutated genes in
    specific cancer based on mutation signature (SBS) using 5 fold cross validation

"""

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, auc
import seaborn as sns

import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('..', 'my_utilities')))
from my_utilities import my_config as cfg
from my_utilities import my_model as my_model
from my_utilities import my_tools as tool

import warnings

warnings.filterwarnings('ignore')


# implement the function of drawing the roc and auc graph
def roc_draw(y_t, y_p, title, cancer___type, gene_lst):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # draw it for all of the label
    n_classes = y_t.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_t[:, i], y_p[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    lg = 0
    # Plot of a ROC curve for a specific class
    plt.figure()
    for j in range(n_classes):
        plt.plot(fpr[j], tpr[j], label='ROC curve ' + gene_lst[j] + ' (area = %0.2f)' % roc_auc[j])
        lg = plt.legend(bbox_to_anchor=(0.5, 0.5), loc='upper left', prop={'size': 6})
        plt.tight_layout()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for gene in cancer %s' % cancer___type)
    plt.legend(loc="lower right", prop={'size': 6})
    if not os.path.exists('./result/gene_classification_roc_auc'):
        os.makedirs('./result/gene_classification_roc_auc')
    plt.savefig(
        './result/gene_classification_roc_auc/The_roc_auc_for_validation_in_fold_{0}_for_class_{1}.png'.format(
            title, cancer___type), dpi=300,
        format='png',
        bbox_extra_artists=(lg,),
        bbox_inches='tight')
    plt.close()


# process the data for specific cancer class
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


# the function to obtain the training_x,testing_x,training_y and testing_y
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


def train(train_x, train_y, test_x, test_y, fold):
    x_train = torch.tensor(train_x, dtype=torch.float32)
    y_train = torch.tensor(train_y, dtype=torch.float32)
    batch_size = cfg.BATCH_SIZE
    batch_count = int(len(x_train) / batch_size) + 1

    save_data = [['epoch', 'loss', 'train accuracy']]
    for epoch in range(cfg.GENE_EPOCH):
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

        print("Epoch: {}, Loss: {:.5f}, Train Accuracy: {:.5f}".
              format(epoch, epoch_loss / batch_count, acc / batch_count))
        save_data.append([epoch, epoch_loss / batch_count, acc / batch_count])
    acc_test = score(test_x, test_y)
    print("The cross-validation test accuracy on fold " + str(fold) + " is :", acc_test)
    return acc_test


# score the classification accuracy for each gene in each cancer and draw the roc graph
def score(test_x, test_y, title=0, cancer__type="", gene_list=None, gene_list_mutation_prob=None, final=False):
    model.eval()
    x_test = torch.tensor(test_x, dtype=torch.float32)
    y_test = torch.tensor(test_y, dtype=torch.float32)

    y_pred = model(x_test)

    y_pred_for_roc = y_pred.detach().numpy()

    y_pred = y_pred.detach().numpy()
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0

    acc_test = np.mean(np.sum((y_test.detach().numpy() - y_pred) == 0, axis=0) / y_test.detach().numpy().shape[0])
    if final:
        roc_draw(y_test, y_pred_for_roc, title, cancer__type, gene_list)
        tool.gene_class_report(y_test.detach().numpy(), y_pred, cancer__type, title, gene_list, gene_list_mutation_prob)
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

    # load the gene occurrence probability in each cancer
    gene_prob = pd.read_csv('../statistics/gene_distribution/gene_prob.csv')
    cancer_prob = {}
    for name, item in gene_prob.groupby('cancer type'):
        cancer_prob[name] = item

    # performing the 5 fold cross validation
    for fold in range(cfg.CROSS_VALIDATION_COUNT - 1):

        for cancer_type in cfg.ORGAN_NAMES:

            gene_list_for_cancer = []
            gene_freq_list_for_cancer = []

            gene_list_final_for_cancer = []
            gene_freq_list_final_for_cancer = []

            for gene in cfg.GENE_NAMES:
                gene_list_for_cancer.append((gene, cancer_prob[cancer_type][gene].values[0]))
                gene_freq_list_for_cancer.append(cancer_prob[cancer_type][gene].values[0])

            # find the top 5 gene's index in pandas frame
            top_5_index = list(reversed(
                sorted(range(len(gene_freq_list_for_cancer)), key=lambda i: gene_freq_list_for_cancer[i])[-5:]))

            # find those gene and their freq as (gene,freq)
            res_list = [gene_list_for_cancer[i] for i in top_5_index]

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

            test_acc.append(train(train_x, train_y, test_x, test_y, fold))
            valid_acc.append(
                score(valid_x, valid_y, title=fold, cancer__type=cancer_type, gene_list=gene_list_final_for_cancer,
                      gene_list_mutation_prob=gene_freq_list_final_for_cancer, final=True))
            print('The %d fold，The best testing accuracy for trained model for %s at this fold is %.4f，the validation '
                  'accuracy '
                  'for this fold is %.4f' % (fold, cancer_type, test_acc[-1], valid_acc[-1]))

            # save the weight of each sbs for each highly frequented gene in that cancer
            weight = model.layer.weight.detach().numpy()
            bias = model.layer.bias.detach().numpy()
            if not os.path.exists('./result'):
                os.makedirs('./result')

            np.save("./result/gene_sbs_weights/gene_type-weight_in_fold%d_for_%s.npy" % (fold, cancer_type), weight)
            np.save("./result/gene_sbs_weights/gene_type-bias_in_fold%d_for_%s.npy" % (fold, cancer_type), bias)
            print('save weight file to ./result')
        test_acc_fold.append(np.mean(test_acc))
        valid_acc_fold.append(np.mean(valid_acc))

    # save the classification result in each fold to log file for observation
    with open('./result/gene_generalized_accuracy/5_fold_accuracy_for_test_data.txt', 'w') as f:
        for item_i in range(len(test_acc_fold)):
            f.write("The fold %d accuracy : %s\n" % (item_i + 1, test_acc_fold[item_i]))

    with open('./result/gene_generalized_accuracy/5_fold_accuracy_for_validation_data.txt', 'w') as f:
        for item_j in range(len(valid_acc_fold)):
            f.write("The fold %d accuracy : %s\n" % (item_j + 1, valid_acc_fold[item_j]))

    print('The 5 fold cross validation has 5 testing across all 32 cancers result,they are :', test_acc_fold)
    print('The validation accuracies for 5 fold cross validation across all 32 cancers result,they are :',
          valid_acc_fold)
