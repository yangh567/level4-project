"""

This file is used to provide all of the models built

"""

import torch
from torch import nn
import torch.nn.functional as F
from keras.layers import Dense, Activation, Flatten, Dropout, Conv1D
from keras.models import Sequential
from keras.utils.vis_utils import plot_model


# constructing the Backpropagation network for classification_cancer_analysis
class SoftMaxBPNet(nn.Module):
    def __init__(self, feature_num, class_num):
        super(SoftMaxBPNet, self).__init__()
        self.feature_num = feature_num
        self.cls_num = class_num
        self.layer = nn.Linear(feature_num, class_num)

    # the forward action is performed by softmax
    def forward(self, x):
        x = self.layer(x)
        x = torch.softmax(x, dim=1)
        return x


# the function used to return the model built
def complex_cnn_model(num_features):
    complex_model = Sequential()
    # first conv layer
    complex_model.add(Conv1D(8, kernel_size=3, padding='same', input_shape=(num_features, 1)))
    complex_model.add(Activation('tanh'))
    complex_model.add(Dense(2))
    # second conv layer
    complex_model.add(Conv1D(8, kernel_size=2, strides=1, padding='same'))
    # third conv layer
    complex_model.add(Conv1D(16, kernel_size=2, strides=1, padding='same'))
    complex_model.add(Activation('tanh'))
    complex_model.add(Dense(2))
    # fourth conv layer
    complex_model.add(Conv1D(16, kernel_size=2, strides=1, padding='same'))
    complex_model.add(Activation('tanh'))
    # fifth conv layer
    complex_model.add(Conv1D(32, kernel_size=2, strides=1, padding='same'))
    complex_model.add(Activation('tanh'))
    # flatten layer
    complex_model.add(Flatten())
    complex_model.add(Activation('tanh'))
    complex_model.add(Dropout(0.5))
    complex_model.add(Dense(2))
    # output layer
    complex_model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # plot the model.uncomment until you installed pydot and graphviz
    # plot_model(complex_model, to_file='./result/complex_cnn_model_plot.pdf', show_shapes=True, show_layer_names=True)

    return complex_model


# the function used to return the model built
def simple_cnn_model(num_features):
    simple_model = Sequential()
    # first conv layer
    simple_model.add(Conv1D(filters=8, kernel_size=3, padding='SAME', input_shape=(num_features, 1)))
    simple_model.add(Activation('tanh'))
    # second conv layer
    simple_model.add(Conv1D(16, kernel_size=3, strides=1, padding='same'))
    # flatten layer
    simple_model.add(Flatten())
    simple_model.add(Activation('tanh'))
    # regularization
    simple_model.add(Dropout(rate=0.5))
    # fully connected layer
    simple_model.add(Dense(2))
    # output layer
    simple_model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    # # plot the model.uncomment until you installed pydot and graphviz
    # plot_model(model, to_file='./result/simple_cnn_model_plot.pdf', show_shapes=True, show_layer_names=True)

    return simple_model
