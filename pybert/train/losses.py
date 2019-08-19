#encoding:utf-8
from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# I refered https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py

class FocalLoss(nn.modules.loss._WeightedLoss):

    def __init__(self, gamma=2, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='none', balance_param=1):
        super(FocalLoss, self).__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, output, target):
        # inputs and targets are assumed to be BatchxClasses
        assert len(output.shape) == len(target.shape)
        assert output.size(0) == target.size(0)
        assert output.size(1) == target.size(1)

        weight = Variable(self.weight)

        # compute the negative likelyhood
        if self.weight:
            logpt = - F.binary_cross_entropy_with_logits(output, target, pos_weight=weight, reduction=self.reduction)
        else:
            logpt = - F.binary_cross_entropy_with_logits(input = output, target = target, reduction=self.reduction)
        pt = torch.exp(logpt)

        '''target_one = target
        target_one_sum = target_one.sum(1)
        target_zero = torch.ones(target.shape)-target
        target_zero_sum = target_zero.sum(1)
        alpha_one = target_one*target_zero_sum[:,None]/target.shape[1]
        alpha_zero = target_zero*target_one_sum[:, None]/target.shape[1]
        alpha = alpha_one+alpha_zero'''
        # compute the loss
        #focal_loss = -((1 - pt) ** self.gamma) * logpt
        #balanced_focal_loss = self.balance_param * focal_loss
        balanced_focal_loss = (-((1 - pt) ** self.gamma) * logpt).mean()
        return balanced_focal_loss

__call__ = ['CrossEntropy','BCEWithLogLoss']

class CrossEntropy(object):
    def __init__(self):
        self.loss_f = CrossEntropyLoss()

    def __call__(self, output, target):
        loss = self.loss_f(input=output, target=target)
        return loss

class BCEWithLogLoss(object):
    def __init__(self):
        self.loss_fn = BCEWithLogitsLoss()

    def __call__(self,output,target):
        loss = self.loss_fn(input = output,target = target)
        return loss
