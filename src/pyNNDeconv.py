# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt

from pyNN import *


# Model initialization

model = DecCNN()


# Data input and configuration

while True:
    file_exp = input('Please, enter the experimental data file... ')
    try:
        raw_data = np.genfromtxt(
            file_exp,
            skip_header=0,
            usecols=(0, 1)
            )

        break
    except FileNotFoundError as exc:
        print(exc)

while True:
    file_weights = input('Please, enter the model weights file... ')
    try:
        model.load_state_dict(torch.load(file_weights, map_location=torch.device('cpu')))
        break
    except FileNotFoundError as exc:
        print(exc)

while True:
    file_out = input('Please, enter the output filename... ')
    if file_out:
        break


# CNN deconvolution

norm_factor = raw_data[:, 1].max()
raw_data[:, 1] /= norm_factor

x_trunc = torch.tensor(raw_data[:5000, 1], dtype=torch.float).unsqueeze(0).unsqueeze(0)
y_trunc = target_func_inv(
    model(x_trunc).detach().squeeze(0).squeeze(0).numpy()
)
y_trunc *= norm_factor

out_data = np.vstack(
    (
        raw_data[:5000, 0],
        y_trunc
    )
).T


# Data saving and plotting

np.savetxt(
    file_out,
    out_data
)

plt.plot(raw_data[:, 0], raw_data[:, 1], color='red', label='raw data')
plt.plot(raw_data[:5000, 0], y_trunc, color='navy', label='CNN-ed data')
plt.legend()
plt.show()