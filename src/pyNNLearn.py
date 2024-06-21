# -*- coding: utf-8 -*-


import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

from pyCore import CustomDataset
from pyNN import *


NPROCS = 0

NORM_FACTOR = 1.0
POISSON_INT_SCALE = 40000.0
POISSON_PROBABILITY = 0.75

NUM_EPOCHS = 2500
BATCH_SIZE = 1000
PATCH_SIZE = 5000
PATCH_INTERSECT = 1000
RESOLUTION_MULT = 1

LEARNING_RATE = 0.0001
NOISE_REG_ALPHA = 0.4
SCHED_FACTOR = 0.5
SCHED_PATIENCE = 10
LOG_EVERY_N_STEP = 1000


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cuda available: {torch.cuda.is_available()} \n")

model = DecCNN().to(device)
transform = transforms.RandomApply(
    [
        PoissonNoiseAug(intensity=POISSON_INT_SCALE)
    ],
    p=POISSON_PROBABILITY
)

criterion = CustomNNLoss(alpha=NOISE_REG_ALPHA)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=SCHED_FACTOR,
    patience=SCHED_PATIENCE,
    verbose=True
    )


def train_loop(train_loader, model, criterion, optimizer, scheduler):

    num_batches = len(train_loader)
    train_loss = 0.0

    for x, y in train_loader:
        
        pos = np.random.randint(0, x.shape[2]-PATCH_SIZE-1)

        x_trunc = x[:, :, pos:(pos+PATCH_SIZE)]
        y_trunc = y[:, pos*RESOLUTION_MULT:(pos*RESOLUTION_MULT)+PATCH_SIZE*RESOLUTION_MULT]

        y_pred_trunc = model(x_trunc.to(device))
        loss = criterion(y_pred_trunc, y_trunc.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    train_loss /= num_batches
    print(f'Train loss: {train_loss:.8f}')

    if scheduler:
        scheduler.step(train_loss)

    return train_loss


def val_loop(val_loader, model, criterion):
    
    num_batches = len(val_loader)
    val_loss = 0.0

    pred, true = [], []

    with torch.no_grad():
        for x, y in val_loader:
            
            pos = 0

            x_trunc = x[:, :, pos:(pos+PATCH_SIZE)]
            y_trunc = y[:, pos*RESOLUTION_MULT:(pos*RESOLUTION_MULT)+PATCH_SIZE*RESOLUTION_MULT]

            y_pred_trunc = model(x_trunc.to(device))
            loss = criterion(y_pred_trunc, y_trunc.to(device))
            val_loss += loss.item()

            pred.extend(y_pred_trunc.cpu().detach().squeeze(0).squeeze(0).numpy())
            true.extend(y_trunc.cpu().detach().squeeze(0).squeeze(0).numpy())
    
    val_loss /= num_batches
    r2_metrics = r2_score(true, pred)

    print(f'Val loss: {val_loss:.8f}')
    print(f'Val R2 score: {r2_metrics:.8f}')
    return val_loss, r2_metrics


# User input

in_weights = input('Please, specify the input weights file (<Enter> for none)... ')
if in_weights:
    model.load_state_dict(torch.load(in_weights))
folder_out = input('Please, specfy the name of the data folder... ')
file_out = input('Please, specify the output filename prefix... ')

# Data Loading

data, target = [], []

for file in os.listdir(f'{folder_out}/'):

    if 'data_' not in file:
        continue

    dataset = np.genfromtxt(
        f'{folder_out}/{file}',
        skip_header=1
    )
    norm_den = dataset[::RESOLUTION_MULT, 0].max() * NORM_FACTOR

    data.append(dataset[::RESOLUTION_MULT, 0] / norm_den)
    target.append(target_func(dataset[:, 1] / norm_den))

data = np.array(data)
target = np.array(target)

data = torch.Tensor(
    data
).unsqueeze(1)

target = torch.Tensor(
    target
)

print(data.shape)
print(target.shape)

x_train, x_val, y_train, y_val = train_test_split(
    data,
    target,
    train_size=0.8,
    shuffle=True
)
x_val, x_test, y_val, y_test = train_test_split(
    x_val,
    y_val,
    train_size=0.5,
    shuffle=True
)

train_dataset = CustomDataset(x_train, y_train, transform=transform)
val_dataset = CustomDataset(x_val, y_val)
test_dataset = CustomDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NPROCS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NPROCS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NPROCS)


# Learning Sequence

train_hist, val_hist, r2_val_hist = [], [], []

for epoch in range(NUM_EPOCHS):

    print(f'Epoch #{epoch}')

    train_loss = train_loop(train_loader, model, criterion, optimizer, scheduler)
    val_loss, r2_metric = val_loop(val_loader, model, criterion)

    train_hist.append(train_loss)    
    val_hist.append(val_loss)
    r2_val_hist.append(r2_metric)

    if not epoch % LOG_EVERY_N_STEP and epoch:
        torch.save(model.state_dict(), f'weights_epoch{epoch}.pth')


torch.save(model.state_dict(), f'{file_out}.pth')

history = np.array(
    [[i for i in range(NUM_EPOCHS)], train_hist, val_hist, r2_val_hist]
).T
np.savetxt(
    f'{file_out}_log.txt',
    history,
    header='Epoch# train_loss val_loss r2_val_metrics'
)

plt.plot(history[:, 0], history[:, 1], color='red', label='train_loss')
plt.plot(history[:, 0], history[:, 2], color='navy', label='val_loss')
plt.legend()
plt.show()
