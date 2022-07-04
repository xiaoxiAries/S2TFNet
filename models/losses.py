#!/usr/bin/env python3

"""Loss functions."""

import torch.nn as nn
import torch


def loss_func(cfg):
    loss_name = cfg.MODEL.LOSS_FUNC
    if loss_name == 'cross_entropy':
        loss_fun = CE()
    elif loss_name == 'CE_L1':
        loss_fun = CE_L1(cfg)
    elif loss_name == 'CE_L1_L2':
        loss_fun = CE_L1_L2(cfg)
    return loss_fun


class CE(nn.Module):
    def __init__(self):
        super(CE, self).__init__()
        self.CEloss_func = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, inputs, target):
        loss = self.CEloss_func(inputs[0], target).cuda()
        return loss


class CE_L1(nn.Module):
    def __init__(self, cfg):
        super(CE_L1, self).__init__()
        self.lamda = cfg.MODEL.LAMDA
        self.CEloss_func = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, inputs, target):
        CEloss = self.CEloss_func(inputs[0], target).cuda()
        saliency = torch.norm(inputs[1], p=1).cuda()
        loss = CEloss + self.lamda * saliency
        loss = loss.cuda()
        return loss


class CE_L1_L2(nn.Module):
    def __init__(self, cfg):
        super(CE_L1_L2, self).__init__()
        self.lamda = cfg.MODEL.LAMDA
        self.gama = cfg.MODEL.GAMA
        self.theta = cfg.MODEL.THETA
        self.CEloss_func = nn.CrossEntropyLoss(reduction="mean")
        self.MSEloss = nn.MSELoss(size_average=True, reduce=True)

    def forward(self, inputs, target):
        inputs[2][0] = torch.cat(inputs[2][0], dim=2)
        inputs[3][0] = torch.cat(inputs[3][0], dim=2)
        CEloss = self.CEloss_func(inputs[0], target).cuda()
        saliency = torch.norm(inputs[1], p=1).cuda()
        TSFP_loss1 = self.MSEloss(inputs[2][0], inputs[2][1]).cuda()
        TSFP_loss2 = self.MSEloss(inputs[3][0], inputs[3][1]).cuda()
        loss = CEloss + self.lamda * saliency + self.gama * TSFP_loss1 + self.theta * TSFP_loss2
        loss = loss.cuda()
        return loss
