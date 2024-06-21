# -*- coding: utf-8 -*-


''' Synthetic XRD generation
     Voigt Bragg peaks and Gaussian diffuse halo
     More realistics FWHMs of Bragg peaks are implied
     Target profiles are additionally blurred by a gaussian function (TARGET_BLUR_SIGMA=0.04)
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.special import voigt_profile
from scipy.signal import fftconvolve

from pyCore import XRDConv


CUKA1 = 1.540598
CUKA2 = 1.544426
MOKA1 = 0.709317
MOKA2 = 0.713607

KA1_MULT = 0.5771816010184

DEFAULT_TWOTHETA_STEP = 0.01

KERNEL_SIZE = 301 # DEFAULT : 3001
SUBKERNEL_SIZE = 301
CUTOFF = 500

BLUR_FUNCTION_FILE = 'LaB6/PDF_test/LaB6_2_relevant.pth'
BLUR_TARGET_SIGMA = 0.04

PEAKS_NUM_MIN = 10
PEAKS_NUM_MAX = 300
VOIGT_GAUSS_MIN = 0.00
VOIGT_GAUSS_SIGMA = 0.025
VOIGT_LORENTZ_MIN = 0.005
VOIGT_LORENTZ_SIGMA = 0.025
INT_SPREAD = 1.0

DIFFUSE_NUM_MIN = 0
DIFFUSE_NUM_MAX = 25
DIFFUSE_FWHM_MIN = 10.0
DIFFUSE_FWHM_MAX = 50.0
DIFFUSE_INT_SPREAD = 10.0

NUM_DATASETS = 250


def lorentz(x, mu=0.0, sigma=1.0):
    return 1.0 / (1.0 + ((x-mu)/sigma)**2) / sigma / np.pi

def gauss(x, mu=0.0, sigma=1.0):
    return 1/sigma/np.sqrt(2*np.pi) * np.exp(-(x-mu)**2/2/sigma**2)

def voigt_peak(x, pos, sigma, gamma):
    return voigt_profile(x - pos, sigma, gamma)


x = np.linspace(0.05-CUTOFF*DEFAULT_TWOTHETA_STEP, 147.35+CUTOFF*DEFAULT_TWOTHETA_STEP, 14731+2*CUTOFF)  # 2th step 0.01
y = np.zeros(14731+2*CUTOFF)

xrd_gen = XRDConv(kernel_size=KERNEL_SIZE, subkernel_size=SUBKERNEL_SIZE, x_data=x, doublet_enabled=False)
xrd_gen.load_state_dict(torch.load(BLUR_FUNCTION_FILE))

x_target_blur = np.arange(-10.00, 10.01, 0.01)
target_blur = gauss(x_target_blur, mu=0.0, sigma=BLUR_TARGET_SIGMA)
target_blur_norm = target_blur.sum()


for set in range(NUM_DATASETS):

    y = np.zeros(14731+2*CUTOFF)

    num_peaks = np.random.randint(PEAKS_NUM_MIN, PEAKS_NUM_MAX)
    num_diffuse = np.random.randint(DIFFUSE_NUM_MIN, DIFFUSE_NUM_MAX)

    for peak in range(num_peaks):
        pos = np.random.rand() * 147.30 + 0.05
        sigma = np.abs(np.random.randn()) * VOIGT_GAUSS_SIGMA + VOIGT_GAUSS_MIN
        gamma = np.abs(np.random.randn()) * VOIGT_LORENTZ_SIGMA + VOIGT_LORENTZ_MIN
        int_ = np.random.rand() * INT_SPREAD

        y += voigt_peak(x, pos, sigma, gamma) * int_

    for peak_diff in range(num_diffuse):
        pos = np.random.rand() * 147.30 + 0.05
        sigma = np.random.rand() * (DIFFUSE_FWHM_MAX - DIFFUSE_FWHM_MIN) + DIFFUSE_FWHM_MIN
        int_ = np.random.rand() * DIFFUSE_INT_SPREAD

        y += gauss(x, pos, sigma) * int_
        
    # target gaussian blurring
    y_target_blur = fftconvolve(y, target_blur, mode='same') / target_blur_norm

    # input instrumental blurring
    y_tens = torch.Tensor(y).unsqueeze(0)
    y_blurred = xrd_gen(y_tens).detach().squeeze(0).numpy() * 10.0

    dataset = np.vstack(
        [
            y_blurred[CUTOFF:-CUTOFF],
            y_target_blur[CUTOFF:-CUTOFF]
        ]
    ).T

    np.savetxt(
        f'NNXrd_GW/data_mode01_sig_{BLUR_TARGET_SIGMA}_{set:0>4}.txt',
        dataset,
        delimiter=' ',
        header='x y'
    )

