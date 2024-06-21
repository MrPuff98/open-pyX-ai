# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from pyCore import XRDConv, RLLoss


INST_FACTOR = 1e-6


target_func = lambda x: x #lambda x: np.log(1.0 + x)
target_func_inv = lambda x: x #lambda x: np.exp(x) - 1.0


class DecCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=(27,),
            #padding='same',
            bias=False
        )

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=(15,),
            #padding='same',
            bias=False
        )

        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=(9,),
            #padding='same',
            bias=False
        )

        self.conv4 = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=(3,),
            #padding='same',
            bias=False
        )

        self.conv5 = nn.Conv1d(
            in_channels=256,
            out_channels=512,
            kernel_size=(3,),
            #padding='same',
            bias=False
        )

        self.lin1 = nn.Linear(
            in_features=9728,
            out_features=4096,
            bias=False
        )

        self.lin2 = nn.Linear(
            in_features=4096,
            out_features=3000,
            bias=False
        )

        self.lin3 = nn.Linear(
            in_features=3000,
            out_features=5000,
            bias=False
        )

        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=(3,))
        self.flatten = nn.Flatten()

    def forward(self, data):

        out = self.conv1(data)
        out = self.pool(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.pool(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.pool(out)
        out = self.act(out)

        out = self.conv4(out)
        out = self.pool(out)
        out = self.act(out)

        out = self.conv5(out)
        out = self.pool(out)
        out = self.act(out)

        out = self.flatten(out)

        out = self.lin1(out)
        out = self.act(out)

        out = self.lin2(out)
        out = self.act(out)

        out = self.lin3(out)

        return out
    

class WeightSqrtMSELoss(nn.Module):

    def __init__(self, inst_factor):
        super().__init__()
        self.inst_factor = inst_factor

    def forward(self, output, target):
        wght = torch.sqrt(target) + self.inst_factor
        mse_w = (wght * (output-target)**2).sum(axis=1) / wght.sum(axis=1)
        return mse_w.mean()


class WeightGradMSELoss(nn.Module):

    def __init__(self, inst_factor):
        super().__init__()
        self.inst_factor = inst_factor

    def forward(self, output, target):
        
        diff = output - target
        grad = torch.abs(torch.gradient(target, dim=-1)[0])

        wght = grad + self.inst_factor
        wght /= wght.mean()

        loss = torch.mean(wght * (diff ** 2))
        return loss


class CustomNNLoss(nn.Module):

    def __init__(self, alpha):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha

    def forward(self, output, target):
        loss = self.mse_loss(output, target)

        if self.alpha:
            loss += self.noise_penalty(output) * self.alpha

        return loss
    
    def noise_penalty(self, output):
        penalty = ((output[:, :-2] + output[:, 2:] - 2*output[:, 1:-1])**2).mean()
        return penalty


class PoissonNoiseAug():

    def __init__(self, intensity=1.0):
        self.intensity = intensity

    def __call__(self, data):
        noise = np.random.poisson(data.numpy() * self.intensity) / self.intensity
        noisy_data = torch.tensor(noise, dtype=data.dtype)
        return noisy_data
