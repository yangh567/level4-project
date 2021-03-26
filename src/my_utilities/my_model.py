"""

This file is used to provide the BackPropagation strategy for the model for
taking weight for all sbs signatures for each cancers and gene

"""

import torch
from torch import nn
import torch.nn.functional as F

# constructing the Backpropagation network for classification_cancer_analysis
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
        return x


# constructing the multiple label element based Backpropagation network for classification_cancer_analysis(DEPRECATED)
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


# the loss that can help with class imbalance(DEPRECATED)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha])
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()
