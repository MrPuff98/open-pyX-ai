# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

from pyCore import XRDConv, Preprocessor, CustomDataset, MSEPenalized


CUKA1 = 1.540598
CUKA2 = 1.544426
MOKA1 = 0.709317
MOKA2 = 0.713607

KA1_MULT = 0.5771816010184

DEFAULT_L1 = 10.0
DEFAULT_TWOTHETA_STEP = 0.001
DEFAULT_SUBFILTER_SIZE = 201
DEFAULT_FILTER_SIZE = 2001
INIT_WEIGHT_SCALE = 0.01
BATCH_SIZE = 5000      
LEARNING_RATE = 0.005   
NUM_EPOCHS = 251
SHUFFLE_ENABLED = True


# Data input and configuration

while True:
    file_data = input('Please, enter the reference data file... ')
    try:
        data = np.genfromtxt(
            file_data,
            skip_header=1,
            usecols=(0, 1)
            )
        break
    except FileNotFoundError as exc:
        print(exc)

while True:
    file_blurred = input('Please, enter the blurred data file... ')
    try:
        target = np.genfromtxt(
            file_blurred,
            skip_header=1,
            usecols=(0, 1)
            )
        break
    except FileNotFoundError as exc:
        print(exc)

try:
    lambda1, lambda2 = tuple(map(np.float64, input('Please, enter the Ka12 wavelengths (<Enter> for default)').split()))
except:
    lambda1, lambda2 = CUKA1, CUKA2

try:
    kernel_size = int(input('Please, enter the filter kernel size (<Enter> for default)'))
except ValueError:
    print(f'Kernel size set to {DEFAULT_FILTER_SIZE}')
    kernel_size = DEFAULT_FILTER_SIZE
    subkernel_size = DEFAULT_SUBFILTER_SIZE

while True:
    try:
        subkernel_size = int(input('Please, enter the reduced filter size (<Enter> for default)'))
        break
    except ValueError:
        pass

twotheta_step = data[1, 0] - data[0, 0]

print(
    '#############################',
    'Following parameters entered:',
    f'Reference data file: {file_data}',
    f'Blurred data file: {file_blurred}',
    f'TwoTheta step (deg): {twotheta_step}',
    f'Ka12 wavelengths: {lambda1} {lambda2}',
    f'Filter kernel size: {kernel_size}',
    f'Subfilter kernel size {subkernel_size}',
    '#############################\n',
    sep='\n'
    )
input('Press <Enter> to continue')

# Data preprocesing and model initialization

model = XRDConv(kernel_size=kernel_size, subkernel_size=subkernel_size, x_data=data[:, 0], doublet_enabled=False)
preprocessor = Preprocessor(lambda1=lambda1, lambda2=lambda2, twotheta_step=twotheta_step)

target[:, 1] = preprocessor.norm_max(target[:, 1])
data[:, 1] = preprocessor.norm_int(data[:, 1], target[:, 1])
#data_doublet = preprocessor.doublet_fast(data)             # DOUBLET DISABLED!!!

x = torch.tensor(data[:, 1]).float()                        # DOUBLET DISABLED!!! 
y = torch.tensor(target[:, 1]).float()

data_train = CustomDataset(x.clone().detach().unsqueeze(0), y.clone().detach().unsqueeze(0))
train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=SHUFFLE_ENABLED)

criterion = MSEPenalized(L1=DEFAULT_L1)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning sequence

print('Learning sequence initiated...')

num_epochs = NUM_EPOCHS
loss_hist = []
for epoch in range(num_epochs):
    print(f'Epoch {epoch} begins...')
    hist_loss = 0
    for i, batch in enumerate(train_loader, 0):
        print('.', end='')
        data_batch, labels = batch
        optimizer.zero_grad()
        pred = model(data_batch)
        loss = criterion(pred, labels, model.filter_instr.weight)
        loss.backward()
        optimizer.step()
        hist_loss += loss.item()

    loss_hist.append(hist_loss / len(train_loader))
    print(f'\nEpoch {epoch} results: penalized loss={loss_hist[epoch]:.4e}...')
    if not epoch % 50:
        with open(f'{file_data}.log', 'a') as log_file:
            print(f'Epoch {epoch} weights:\n', *model.filter_instr.weight.detach().numpy().flatten(), '\n', file=log_file)

print('Learning successfully finished...')


# Results output and weights saving

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)

prof_function = model.filter_instr.weight.detach().numpy()

file_weights = ''

while True:
    try:
        while not file_weights:
            file_weights = input('Please, enter the output filename...')
        torch.save(model.state_dict(), file_weights)
        print('File saved')
        break
    except:
        print('Sorry, an error has occured...')

# Plotting

fig, ax = plt.subplots(1, 2)
ax[0].set_title('Perfect and spoilt data')
ax[0].plot(data[:, 0], data[:, 1], 'r-')
#ax[0].plot(data_doublet[:, 0], data_doublet[:, 1], 'g-')      # DOUBLET DISABLED
ax[0].plot(target[:, 0], target[:, 1], 'b-')
ax[0].plot(
    data[:, 0],
    model(x.clone().detach().unsqueeze(0)).detach().squeeze(0).numpy(),
    'g-'
    )

ax[1].set_title('Predicted profile function')
ax[1].plot(prof_function, 'r-')

plt.show()