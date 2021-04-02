"""
This file provides the tools to analyse on the gene distribution in specific cancer and their classification accuracies(DEPRECATED)

"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
import my_config as cfg

# the weight of each class in averaging the classification accuracy of gene
weight_lst = [162, 98, 90, 87, 85, 84, 80, 80, 74, 73, 67, 67, 64, 60, 59, 50, 48, 43, 34, 34, 31, 29, 27, 25,
              21, 19, 19, 18, 15, 13, 9, 8]


# this function is used to plot the total summary loss and total summary accuracy in each fold for gene classification
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
    axs[1].legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', prop={'size': 6})

    plt.tight_layout()
    plt.savefig(
        './result/gene_classification_converge/The_convergence_graph_in_fold_%d.png' % title, dpi=300,
        format='png',
        bbox_inches='tight')
    plt.close()


# the function to save the accuracy results of gene in each cancers in a fold for gene classification
def save_accuracy_results(fold, cancer_list, cancer_gene, validation_acc, cancer_gene_freq):
    data = {
        'cancer_type': cancer_list,
        'gene_name': cancer_gene,
        'Accuracy': validation_acc,
        'Mutation_frequency': cancer_gene_freq
    }
    # save as pandas dataframe and save to file
    df = pd.DataFrame(data)
    df.to_csv('./result/gene_classification_accuracy/The_classification_across_gene_in_fold_%d.csv' % (
        fold))


# implement the function of drawing the roc and auc graph for gene classification
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
    plt.title('Receiver operating characteristic for gene in all cancers')
    if not os.path.exists('./result/gene_classification_roc_auc'):
        os.makedirs('./result/gene_classification_roc_auc')
    plt.savefig(
        './result/gene_classification_roc_auc/The_roc_auc_for_validation_in_fold_{0}.png'.format(
            title), dpi=300,
        format='png',
        bbox_extra_artists=(lg,),
        bbox_inches='tight')

    plt.close()


# score the classification accuracy for each gene in each cancer
def score(cnn_model, test_x, test_y):
    y_pred = cnn_model.predict(test_x)

    y_c_pred = deepcopy(y_pred)

    y_c_pred[y_c_pred > 0.5] = 1
    y_c_pred[y_c_pred <= 0.5] = 0

    acc_test = np.mean(np.sum((test_y - y_c_pred) == 0, axis=0) / test_y.shape[0])

    return y_pred, acc_test


# the function to find the top driver gene in each cancer in overall data
def find_top_gene(cancer_type, caner_probability, driver_gene_in_c=None, driver_gene_freq_in_c=None):
    # the list used to contain all of the driver gene in that cancer
    gene_list_for_cancer = []
    # the list used to contain all of the driver gene's frequency in that cancer
    gene_freq_list_for_cancer = []
    # the list used to contain the top driver gene in that cancer
    gene_list_final_for_cancer = []
    # the list used to contain the top driver gene's frequency in that cancer
    gene_freq_list_final_for_cancer = []
    # we leave the list extension here to find the top frequently mutated gene if there is more
    for gene in cfg.GENE_NAMES_DICT[cfg.ORGAN_NAMES[cancer_type]]:
        gene_list_for_cancer.append((gene, caner_probability[cfg.ORGAN_NAMES[cancer_type]][gene].values[0]))
        gene_freq_list_for_cancer.append(caner_probability[cfg.ORGAN_NAMES[cancer_type]][gene].values[0])

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
    if driver_gene_in_c is not None and driver_gene_freq_in_c is not None:
        # here, we append the driver gene's name and cancer name for future visualization in ROC
        driver_gene_in_c.append(gene_list_final_for_cancer[0])
        driver_gene_freq_in_c.append(gene_freq_list_final_for_cancer[0])
        # see what is the driver gene in that cancer
        print(gene_list_final_for_cancer, cfg.ORGAN_NAMES[cancer_type])
        # see the frequency of that driver gene in the cancer
        print(gene_freq_list_final_for_cancer)

    return gene_list_final_for_cancer, driver_gene_in_c, driver_gene_freq_in_c


# the function used to find the top gene in that cancer as well as the top 10 sbs in that cancer
def find_top_10_sbs(fold, cancer_type):
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

    return res_cancer_sbs_weight_list


# The function to load the gene occurrence probability in each cancer
def obtain_gene_prob_cancer():
    # load the gene occurrence probability in each cancer
    gene_prob = pd.read_csv('../statistics/gene_distribution/gene_prob.csv')
    cancer_prob = {}
    for name, item in gene_prob.groupby('cancer type'):
        cancer_prob[name] = item

    return cancer_prob
