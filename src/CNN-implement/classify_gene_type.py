"""

    This file is used to test on the self-build complex CNN model on the classification_cancer_analysis of genes
    based on mutation signature (SBS) using 5 fold cross validation

"""
import os
import sys
import numpy as np
import pandas as pd
from keras.optimizers import RMSprop
from sklearn.preprocessing import StandardScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('..', 'my_utilities')))
from my_utilities import my_config as cfg
from my_utilities import my_tools as tool
from my_utilities import my_model as my_model
import warnings

warnings.filterwarnings('ignore')

'''Data Preprocessing'''


# process the data for specific cancer class
def process_data(data, cancer_type, gene_list, sbs_names, scale=True):
    # setting the spatial features to help with constructing cnn
    data_copy = data.copy()
    for sbs_name in cfg.SBS_NAMES:
        # set the sbs that are not important to 0.001*value
        if not sbs_name in sbs_names:
            data_copy[data_copy["organ"] == cancer_type][sbs_name] = 0.001 * \
                                                                     data_copy[data_copy["organ"] == cancer_type][
                                                                         sbs_name]
    # feed the matrix
    x = data_copy[data_copy["organ"] == cancer_type][cfg.SBS_NAMES]
    y = data_copy[data_copy["organ"] == cancer_type][gene_list]
    # if it has mutated multiple times ,when set it as mutated
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


if __name__ == '__main__':
    # read the data
    o_data = []
    for i in range(cfg.CROSS_VALIDATION_COUNT - 1):
        o_data.append(pd.read_csv(os.path.join(cfg.C_V_DATA_PATH, 'cross_validation_%d.csv' % i)))
    valid_dataset = pd.read_csv(os.path.join(cfg.C_V_DATA_PATH, 'validation_dataset.csv'))

    # handling the NaN
    o_data = [item.fillna(0) for item in o_data]
    valid_dataset = valid_dataset.fillna(0)

    # set the recorder to record the trained model's testing accuracy in each fold
    test_acc_fold = []
    valid_acc_fold = []

    # load the gene occurrence probability in each cancer
    gene_prob = pd.read_csv('../statistics/gene_distribution/gene_prob.csv')
    cancer_prob = {}
    for name, item in gene_prob.groupby('cancer type'):
        cancer_prob[name] = item

    '''5 fold cross validation'''
    # performing the 5 fold cross validation
    for fold in range(cfg.CROSS_VALIDATION_COUNT - 1):
        # used to record the total gene classification history in each cancer
        total_gene_history = []
        # the list to append every driver gene's valid_y in each cancer(used for drawing ROC at once)
        all_gene_valid_y = []
        # the list to append every driver gene's valid_pred in each cancer(used for drawing ROC at once)
        all_gene_valid_pred = []
        # the list to append every driver gene's test accuracy in each cancer(used for showing the last 5 fold results)
        test_acc = []
        # the list to append every driver gene's valid accuracy in each cancer(used for showing the last 5 fold results)
        valid_acc = []
        # the list used to store the driver gene in that 32 cancers
        cancer_driver_gene = []
        # cancer driver gene frequency in 32 cancers
        cancer_driver_gene_freq = []

        for cancer_type in range(len(cfg.ORGAN_NAMES)):
            '''Data processing'''
            # find the top frequently mutated gene in that cancer
            gene_list_final, cancer_driver_gene, cancer_driver_gene_freq = tool.find_top_gene(cancer_type, cancer_prob,
                                                                                              cancer_driver_gene,
                                                                                              cancer_driver_gene_freq)
            # find top 10 sbs signatures in that cancer
            top10_sbs_list = tool.find_top_10_sbs(fold, cancer_type)

            # obtain the training data and testing data using those labels and features found above
            train_x, train_y, test_x, test_y = get_data(o_data, fold, cfg.ORGAN_NAMES[cancer_type],
                                                        gene_list_final,
                                                        top10_sbs_list)

            # obtain the validation data using those labels and features found above
            valid_x, valid_y = process_data(valid_dataset, cfg.ORGAN_NAMES[cancer_type], gene_list_final,
                                            top10_sbs_list)
            # record the valid_y for the gene in the cancer for drawing ROC
            all_gene_valid_y.append(valid_y)

            '''Constructing model'''
            # constructing the complex CNN
            n_features = train_x.shape[1]
            model = my_model.complex_cnn_model(n_features)

            # set up optimizer
            rmsp = RMSprop(lr=0.001, rho=0.9)

            # compile the model
            model.compile(loss="binary_crossentropy",
                          optimizer=rmsp,
                          metrics=['acc'])

            # reshape the input data
            x_train = np.expand_dims(train_x, -1)
            x_test = np.expand_dims(test_x, -1)
            x_valid = np.expand_dims(valid_x, -1)

            '''Train the model'''
            # train the model
            history = model.fit(x_train, train_y, epochs=200, batch_size=1280)

            # save the model
            # model.save("./result/my_complex_cnn_model.h5")

            # append the history and the gene
            total_gene_history.append((history, gene_list_final[0]))
            '''Test the model'''
            # test on testing data
            _, accuracy_test = tool.score(model, x_test, test_y)
            '''Evaluate on model'''
            # validate on validation data and store the validation prediction value for later combination of
            # different gene in ROC graph
            valid_y_pred, accuracy_valid = tool.score(model, x_valid, valid_y)
            # store the validation y
            all_gene_valid_pred.append(valid_y_pred)
            # append the testing accuracy and validation accuracy for displaying
            test_acc.append(accuracy_test)
            valid_acc.append(accuracy_valid)
            print("Testing Accuracy: {:.4f}".format(accuracy_test))
            print("Validation Accuracy: {:.4f}".format(accuracy_valid))

        ''' Evaluation here '''
        # Now, we can start evaluation here
        # plot the converge graph for each fold
        tool.plot_epoch_acc_loss(total_gene_history, fold, 200)
        # save the classification result (accuracy) of gene across cancers of this fold to the file
        tool.save_accuracy_results(fold, cfg.ORGAN_NAMES, cancer_driver_gene, valid_acc, cancer_driver_gene_freq)
        # now,we can finally draw the roc for every gene classification in each cancer in this fold
        tool.roc_draw(all_gene_valid_y, all_gene_valid_pred, fold, cancer_driver_gene)
        # we do the weighted averaging calculation here for testing overall classification accuracy
        test_acc_fold.append(
            sum([acc * (weight / sum(tool.weight_lst)) for acc, weight in zip(test_acc, tool.weight_lst)]))
        valid_acc_fold.append(
            sum([acc_1 * (weight_1 / sum(tool.weight_lst)) for acc_1, weight_1 in zip(valid_acc, tool.weight_lst)]))

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
