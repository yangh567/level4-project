"""

    This file is used to test on the self-build model on the classification_cancer_analysis of genes
    based on mutation signature (SBS) using 5 fold cross validation

"""
import os
import sys

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Activation, Flatten, Dropout, Conv1D
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.metrics import roc_curve, auc
from copy import deepcopy
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('..', 'my_utilities')))
from my_utilities import my_config as cfg
from my_utilities import my_tools as tool
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

# the weight of each class in averaging the classification accuracy
weight_lst = [168, 98, 89, 86, 85, 85, 82, 80, 80, 73, 71, 69, 65, 62, 56, 51, 48, 41, 33, 31, 30, 26, 25, 20,
              20,
              17, 15, 13, 12, 11, 9, 7]


# this function is used to plot the total summary loss and total summary accuracy in each fold
def plot_epoch_acc_loss(all_model_history, title, epochs):
    plt.figure(figsize=(3, 4))
    fig, axs = plt.subplots(2)
    fig.suptitle('The convergence of accuracy and loss for gene classification in fold %d' % title)

    total_gene_acc = [0] * epochs
    total_gene_loss = [0] * epochs

    cancer_weight = 0
    for (model_history, gene_name) in all_model_history:
        # summarize history for accuracy
        total_gene_acc = [sum(x) for x in zip(total_gene_acc,
                                              [e * (weight_lst[cancer_weight] / sum(weight_lst)) for e in
                                               model_history.history['acc']])]
        total_gene_loss = [sum(y) for y in zip(total_gene_loss, model_history.history['loss'])]
        cancer_weight += 1

    # there are 32 cancers
    total_gene_loss = [x / 32 for x in total_gene_loss]

    axs[0].plot(total_gene_acc, label="train accuracy")
    # summarize history for loss
    axs[1].plot(total_gene_loss, label="train loss")

    axs[0].set_title('model accuracy')
    axs[0].set_ylabel('accuracy')
    axs[0].set_xlabel('epoch')
    axs[0].legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', prop={'size': 6})
    axs[1].set_title('model loss')
    axs[1].set_ylabel('loss')
    axs[1].set_xlabel('epoch')

    plt.tight_layout()

    plt.savefig(
        './result/gene_classification_converge/The_convergence_graph_in_fold_%d.png' % title, dpi=300,
        format='png',
        bbox_inches='tight')
    plt.close()


# implement the function of drawing the roc and auc graph
def roc_draw(y_t, y_p, title, cancer_driver_gene_list):
    # the total cancer numbers
    cancer_num = len(y_t)
    # the two label(0 : not mutated,1 : mutated)
    n_classes = y_t[0].shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for cancer in range(cancer_num):
        # draw it for all of the label
        for label in range(n_classes):
            fpr[cancer], tpr[cancer], _ = roc_curve(y_t[cancer][:, label], y_p[cancer][:, label])
            roc_auc[cancer] = auc(fpr[cancer], tpr[cancer])

    lg = 0
    # Plot of a ROC curve for a specific class
    plt.figure()
    for cancer_i in range(cancer_num):
        for j in range(n_classes):
            plt.plot(fpr[cancer_i], tpr[cancer_i],
                     label='ROC curve ' + cfg.ORGAN_NAMES[cancer_i] + ':' + cancer_driver_gene_list[
                         cancer_i] + ' (area = %0.2f)' % roc_auc[cancer_i])
            lg = plt.legend(bbox_to_anchor=(1.0, 1.0), loc='best', prop={'size': 6})
            plt.tight_layout()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for gene in all cancers %s in fold' % title)
    if not os.path.exists('./result/gene_classification_roc_auc'):
        os.makedirs('./result/gene_classification_roc_auc')
    plt.savefig(
        './result/gene_classification_roc_auc/The_roc_auc_for_validation_in_fold_{0}.png'.format(
            title), dpi=300,
        format='png',
        bbox_extra_artists=(lg,),
        bbox_inches='tight')

    plt.close()


# process the data for specific cancer class
def process_data(data, cancer_type, gene_list, sbs_names, scale=True):
    print(data)
    x = data[data["organ"] == cancer_type][sbs_names]
    y = data[data["organ"] == cancer_type][gene_list]
    y[y >= 1] = 1
    y[y < 1] = 0
    y = y.values

    if scale:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
    return x, y


# the function to obtain the training_x,testing_x,training_y and testing_y
def get_data(o_data, index, cancer_type, gene_list, sbs_names):
    train = []
    test = None
    for i in range(len(o_data)):
        if i != index:
            train.append(o_data[i])
        else:
            test = o_data[i]
    train = pd.concat(train)
    train_x, train_y = process_data(train, cancer_type, gene_list, sbs_names)
    test_x, test_y = process_data(test, cancer_type, gene_list, sbs_names)
    return train_x, train_y, test_x, test_y


# score the classification accuracy for each gene in each cancer and draw the roc graph
def score(cnn_model, test_x, test_y):
    y_pred = cnn_model.predict(test_x)
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0

    acc_test = np.mean(np.sum((test_y - y_pred) == 0, axis=0) / test_y.shape[0])

    return acc_test


# the focal loss used to solve class imbalance problem
def focal_loss(gamma, alpha):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed


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
        # used to record the total gene classification history in each cancer
        total_gene_history = []
        # the list to append every driver gene's valid_y in each cancer(used for drawing ROC at once)
        all_gene_valid_y = []
        # the list to append every driver gene's valid_pred in each cancer(used for drawing ROC at once)
        all_gene_valid_pred = []
        # the list used to store the cancer type and driver gene in that cancer
        cancer_driver_gene = []
        # we load the weight of each sbs in that cancer
        for cancer_type in range(len(cfg.ORGAN_NAMES)):

            gene_list_for_cancer = []
            gene_freq_list_for_cancer = []

            gene_list_final_for_cancer = []
            gene_freq_list_final_for_cancer = []

            for gene in cfg.GENE_NAMES_DICT[cfg.ORGAN_NAMES[cancer_type]]:
                gene_list_for_cancer.append((gene, cancer_prob[cfg.ORGAN_NAMES[cancer_type]][gene].values[0]))
                gene_freq_list_for_cancer.append(cancer_prob[cfg.ORGAN_NAMES[cancer_type]][gene].values[0])

            # find the top 1 gene's index in pandas frame
            top_1_index = list(reversed(
                sorted(range(len(gene_freq_list_for_cancer)), key=lambda i: gene_freq_list_for_cancer[i])[-1:]))

            # find those gene and their freq as (gene,freq)
            res_list = [gene_list_for_cancer[i] for i in top_1_index]

            # append the gene name into gene_list_final_for_cancer list
            # append the gene mutation frequency to gene_freq_list_final_for_cancer list
            for (a, b) in res_list:
                gene_list_final_for_cancer.append(a)
                gene_freq_list_final_for_cancer.append(b)

            # here, we append the driver gene's name and cancer name for future visualization in ROC
            cancer_driver_gene.append(gene_list_final_for_cancer[0])
            # see what is the driver gene in that cancer
            print(gene_list_final_for_cancer, cfg.ORGAN_NAMES[cancer_type])
            # see the frequency of that driver gene in the cancer
            print(gene_freq_list_final_for_cancer)

            # we load the weight of sbs in that cancer in that fold ,normalize the weights and find the powerful
            # signature in that cancer
            cancer_type_path = '../classification_cancer_analysis/result/cancer_type-weight_' + str(fold) + '.npy'
            cancer_type_weight = np.load(cancer_type_path).T  # shape (49,32)
            cancer_type_scaler = MinMaxScaler()
            cancer_type_nor_weight = cancer_type_scaler.fit_transform(abs(cancer_type_weight))
            # normalize it to 0 and 1
            cancer_type_zero_one_weight = cancer_type_nor_weight / np.sum(cancer_type_nor_weight, axis=0).reshape(1, 32)

            cancer_type_zero_one_weight_c = list(cancer_type_zero_one_weight[:, cancer_type])

            # we find the top 10 weighted sbs signatures comes handy in identify this cancer

            top_10_cancer_sbs_index = list(reversed(
                sorted(range(len(cancer_type_zero_one_weight_c)), key=lambda k: cancer_type_zero_one_weight_c[k])[
                -10:]))

            # get the top 10 sbs signatures' column name(used for feature extraction)
            res_cancer_sbs_weight_list = [cfg.SBS_NAMES[s] for s in top_10_cancer_sbs_index]

            train_x, train_y, test_x, test_y = get_data(o_data, fold, cfg.ORGAN_NAMES[cancer_type],
                                                        gene_list_final_for_cancer,
                                                        res_cancer_sbs_weight_list)

            valid_x, valid_y = process_data(valid_dataset, cfg.ORGAN_NAMES[cancer_type], gene_list_final_for_cancer,
                                            res_cancer_sbs_weight_list)
            # constructing the simple CNN
            n_features = train_x.shape[1]
            model = Sequential()
            model.add(Conv1D(filters=8, kernel_size=3, padding='SAME', input_shape=(n_features, 1)))
            model.add(Activation('tanh'))
            model.add(Conv1D(16, kernel_size=3, strides=1, padding='same'))
            model.add(Flatten())
            model.add(Activation('tanh'))
            model.add(Dropout(rate=0.5))
            model.add(Dense(2))
            model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

            # plot the model.uncomment until you installed pydot and graphviz
            # plot_model(model, to_file='./result/simple_cnn_model_plot.png', show_shapes=True, show_layer_names=True)

            # set up optimizer
            sgd = SGD(lr=0.001, decay=1e-9, momentum=0.9, nesterov=True)
            model.compile(loss=focal_loss(gamma=2., alpha=0.1),
                          optimizer=sgd,
                          metrics=['accuracy'])
            # reshape the input data
            x_train = np.expand_dims(train_x, -1)
            x_test = np.expand_dims(test_x, -1)
            x_valid = np.expand_dims(valid_x, -1)

            # train the model
            history = model.fit(x_train, train_y, epochs=200, batch_size=1000)

            # appending all of the gene's training process in each cancer types in each fold
            total_gene_history.append((history, gene_list_final_for_cancer[0]))

            # test on testing data
            accuracy_test = score(model, x_test, test_y)
            print("Testing Accuracy: {:.4f}".format(accuracy_test))

            # --------------------------------------------------------------------------------------------------------
            # validate on validation data and store the validation prediction value for later combination of ROC graph
            all_gene_valid_y.append(valid_y)

            # store the validation y
            valid_y_pred = model.predict(x_valid)

            valid_y_p = deepcopy(valid_y_pred)

            # store the validation prediction value
            all_gene_valid_pred.append(valid_y_pred)

            # used to collect the classification of each gene in each cancer and stored into csv file under
            # gene_classification_accuracy folder

            valid_y_p[valid_y_p > 0.5] = 1
            valid_y_p[valid_y_p <= 0.5] = 0

            tool.gene_class_report(valid_y, valid_y_p, cfg.ORGAN_NAMES[cancer_type], fold,
                                   gene_list_final_for_cancer, gene_freq_list_final_for_cancer)

            acc_test_valid = np.mean(np.sum((valid_y - valid_y_p) == 0, axis=0) / valid_y.shape[0])

            test_acc.append(accuracy_test)
            valid_acc.append(acc_test_valid)
            # -------------------------------------------------------------------------------------------------------

            print("Validation Accuracy: {:.4f}".format(acc_test_valid))

        # plot the converge graph for each fold
        plot_epoch_acc_loss(total_gene_history, fold, 200)
        # now,we can finally draw the roc for every gene classification in each cancer in this fold
        roc_draw(all_gene_valid_y, all_gene_valid_pred, fold, cancer_driver_gene)
        # we do the weighted averaging calculation here for testing overall classification accuracy
        test_acc_fold.append(sum([acc * (weight / sum(weight_lst)) for acc, weight in zip(test_acc, weight_lst)]))
        valid_acc_fold.append(sum([acc_1 * (weight_1 / sum(weight_lst)) for acc_1, weight_1 in zip(valid_acc, weight_lst)]))

    # save the classification result in each fold to log file for observation
    with open('./result/gene_generalized_accuracy/5_fold_accuracy_for_test_data.txt', 'w') as f:
        for item_i in range(len(test_acc_fold)):
            f.write("The fold %d accuracy : %s\n" % (item_i + 1, test_acc_fold[item_i]))

    with open('./result/gene_generalized_accuracy/5_fold_accuracy_for_validation_data.txt', 'w') as f:
        for item_j in range(len(valid_acc_fold)):
            f.write("The fold %d accuracy : %s\n" % (item_j + 1, valid_acc_fold[item_j]))

    # publish results
    print('The 5 fold cross validation has 5 testing across all 32 cancers result,they are :', test_acc_fold)
    print('The validation accuracies for 5 fold cross validation across all 32 cancers result,they are :',
          valid_acc_fold)
