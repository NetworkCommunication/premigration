#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File : gan.py
# @Project : TP
# @Author : RenJianK
# @Time : 2022/5/19 9:19
import torch.nn as nn

n_hidden = 300


class Generator(nn.Module):
    def __init__(self, n_input=7, n_lane=2, is_g=True):
        super().__init__()
        self.n_input = n_input
        self.g = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU()
        )

        self.classifier = nn.Linear(n_hidden, n_lane)
        self.log_soft = nn.LogSoftmax(dim=1)
        self.normal = nn.Linear(n_hidden, 2)
        self.is_g = is_g

    def forward(self, x):
        g = self.g(x)
        if self.is_g:
            y = self.normal(g)
        else:
            y = self.classifier(g)
            y = self.log_soft(y)
        return y


class Discriminator(nn.Module):
    def __init__(self, n_input=7):
        super().__init__()
        self.d_1 = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, 1),
            nn.ReLU(True)
        )
        self.d_2 = nn.Sequential(
            nn.Linear(2, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, 1),
            nn.ReLU(True)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x, y):
        s_1 = self.d_1(x)
        s_2 = self.d_2(y)
        s = s_1 + s_2
        return self.sig(s)
