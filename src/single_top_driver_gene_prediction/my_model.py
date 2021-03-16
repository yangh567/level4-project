"""

This file is used to provide the BPnet for the model for
taking weight for all sbs signatures for each cancers

"""




import torch
from torch import nn
from torch.nn.functional import softmax

# constructing the Backpropagation network for classification
class BPNet(nn.Module):
    def __init__(self, feature_num, class_num):
        super(BPNet, self).__init__()
        self.feature_num = feature_num
        self.cls_num = class_num
        self.layer = nn.Linear(feature_num, class_num)

    # the forward action is performed by softmax
    def forward(self, x):
        x = self.layer(x)
        x = torch.softmax(x, dim=1)
        # x = torch.sigmoid(x)
        # x = nn.functional.softmax(x)
        return x

# constructing the multiple label element based Backpropagation network for classification
class MultiBPNet(nn.Module):
    def __init__(self, feature_num, class_num):
        super(MultiBPNet, self).__init__()
        self.feature_num = feature_num
        self.cls_num = class_num
        self.layer = nn.Linear(feature_num, class_num)

    def forward(self, x):
        x = self.layer(x)
        x = torch.sigmoid(x)
        return x
