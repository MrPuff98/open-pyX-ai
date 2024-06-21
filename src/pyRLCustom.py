# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

from pyCore import XRDConv, RLLoss


# constants and default parameters initialization

CUKA1 = 1.540598
CUKA2 = 1.544426
MOKA1 = 0.709317
MOKA2 = 0.713607

KA1_MULT = 0.5771816010184

DEFAULT_TWOTHETA_STEP = 0.005
NUM_ITER = 100

KERNEL_SIZE = 3001
SUBKERNEL_SIZE = 301
CUTOFF = 301

# Data input and configuration

while True:
    file_exp = input('Please, enter the experimental data file... ')
    try:
        target = np.genfromtxt(
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
        model = XRDConv(kernel_size=KERNEL_SIZE, subkernel_size=SUBKERNEL_SIZE, x_data=target[:, 0])    # TO BE REWRITTEN
        model.load_state_dict(torch.load(file_weights))
        break
    except FileNotFoundError as exc:
        print(exc)

while True:
    file_out = input('Please, enter the output filename... ')
    if file_out:
        break
        


try:
    lambda1, lambda2 = tuple(map(np.float64, input('Please, enter the Ka12 wavelengths (<Enter> for default)').split()))
except:
    lambda1, lambda2 = CUKA1, CUKA2

twotheta_step = target[1, 0] - target[0, 0]

print(
    '#############################',
    'Following parameters entered:',
    f'Experimental data file: {file_exp}',
    f'Model weights file: {file_weights}',
    f'TwoTheta step (deg): {twotheta_step}',
    f'Ka12 wavelengths: {lambda1} {lambda2}',
    '#############################\n',
    sep='\n'
    )
input('Press <Enter> to continue')


# Tensors initialization

y = torch.tensor(target[:, 1], requires_grad=True).float().unsqueeze(0)
x = torch.tensor(y, requires_grad=True)

criterion = RLLoss()


# Blurring matrix initialization

psf = model.filter_instr.get_interp(
    model.filter_instr.weight
).detach().squeeze(0).squeeze(0).numpy()[::-1]  # Extracting the blurring filter from the model in the numpy format


#unit_blurred = np.zeros(x.shape[-1])
#for i in range(x.shape[-1]):
#    for j in range(
#        max(0, i - KERNEL_SIZE//2),
#        min(x.shape[-1], i + KERNEL_SIZE//2)
#        ):
#        unit_blurred[i] += psf[KERNEL_SIZE//2 - (i - j) - 1]

#np.savetxt(
#    f'{file_out}_HT_unit',
#    unit_blurred.T
#)

#unit_blurred = torch.from_numpy(unit_blurred).float().unsqueeze(0)

with torch.no_grad():
    unit_blurred = torch.flip(model(torch.ones(x.shape).float()), dims=(1,)) # [:, ::-1]    # TO BE CHECKED !!!

np.savetxt(
    f'{file_out}_HT_unit_quick',
    unit_blurred.detach().squeeze(0).numpy()
)


# Deconvolution procedure

print('Deconvolution sequence initiated...')

num_steps = NUM_ITER
loss_hist = []

for step in range(num_steps):
    print(f'Step #{step} begins...')

    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()

    with torch.no_grad():
        lr = x.detach()[:, CUTOFF:-CUTOFF] / unit_blurred.detach()[:, CUTOFF:-CUTOFF]
        x[:, CUTOFF:-CUTOFF] += lr * x.grad[:, CUTOFF:-CUTOFF]
        print(lr, x.grad)
    
    x.grad.zero_()
    model.filter_instr.zero_grad()

    loss_hist.append(loss.item())
    print(f'\nStep {step} results: RLLoss={loss_hist[step]:.4e}')


# Deconvolution results saving

prof_deconv = np.vstack(
    (
        target[:, 0],
        x.detach().squeeze(0).numpy()
    )
).T
best_y_pred = np.vstack(
    (
        target[:, 0],
        model(x).detach().squeeze(0).numpy()
    )
).T

np.savetxt(
    f'{file_out}_deconv_RLCustom_{num_steps}',
    prof_deconv
)
np.savetxt(
    f'{file_out}_best_ypred_RLCustom_{num_steps}',
    best_y_pred
)


# Data visualization

fig, axs = plt.subplots(1, 2)

axs[0].plot(prof_deconv[:, 0], prof_deconv[:, 1], 'r-', label='Deconvoluted Profile')
axs[0].plot(target[:, 0], target[:, 1], 'b-', label='Experimental Profile')

axs[1].plot(best_y_pred[:, 0], best_y_pred[:, 1], 'r-', label='best y_pred')
axs[1].plot(target[:, 0], target[:, 1], 'b-', label='Experimental Profile')
plt.legend()
plt.show()