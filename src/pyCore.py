# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np


CUKA1 = 1.540598
CUKA2 = 1.544426
MOKA1 = 0.709317
MOKA2 = 0.713607

KA1_MULT = 0.5771816010184

DEFAULT_L1 = 0.1
DEFAULT_NEF = 0.0
DEFAULT_INST_FACTOR = 0.0001
DEFAULT_TWOTHETA_STEP = 0.005
DEFAULT_SUBFILTER_SIZE = 200
DEFAULT_LAMBDA1 = 0.70000
DEFAULT_LAMBDA2 = 0.85000
DEFAULT_LAMBDA_SIZE = 16
DEFAULT_FILTER_SIZE = 2000
INSTR_INIT_WEIGHT_SCALE = 0.01
LAMBDA_INIT_WEIGHT_SCALE = 0.01


# Modules implementation


class InterpolatedConv(nn.Module):

    def __init__(
        self,
        in_weight_size=DEFAULT_SUBFILTER_SIZE,
        out_weight_size=DEFAULT_FILTER_SIZE
        ):

        super().__init__()
        self.in_weight_size = in_weight_size
        self.out_weight_size = out_weight_size
        self.weight = nn.parameter.Parameter(torch.randn(in_weight_size)*INSTR_INIT_WEIGHT_SCALE)

    # источник https://github.com/pytorch/pytorch/issues/50334
    def _interp(self, x, xp, fp):
        """One-dimensional linear interpolation for monotonically increasing sample
        points.

        Returns the one-dimensional piecewise linear interpolant to a function with
        given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

        Args:
            x: the :math:`x`-coordinates at which to evaluate the interpolated
                values.
            xp: the :math:`x`-coordinates of the data points, must be increasing.
            fp: the :math:`y`-coordinates of the data points, same length as `xp`.

        Returns:
            the interpolated values, same size as `x`.
        """
        m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
        b = fp[:-1] - (m * xp[:-1])
        indices = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
        indices = torch.clamp(indices, 0, len(m) - 1)
        return m[indices] * x + b[indices]

    def get_interp(self, wgt):
        xp = torch.linspace(0, 1, self.in_weight_size)
        xf = torch.linspace(0, 1, self.out_weight_size)
        weight_int = self._interp(xf, xp, wgt.reshape(-1))
        weight_int = weight_int.reshape((1, 1, -1))
        return weight_int

    def forward(self, x):
        xp = torch.linspace(0, 1, self.in_weight_size)
        xf = torch.linspace(0, 1, self.out_weight_size)
        weight = self._interp(xf, xp, self.weight.reshape(-1))
        weight = weight.reshape((1, 1, -1))
        return nn.functional.conv1d(
            x,
            weight,
            bias=None,
            padding='same'
        )


class LambdaConv(nn.Module):

    def __init__(
        self,
        lambda1=CUKA1,
        lambda2=CUKA2,
        ka1=CUKA1,
        ka2=CUKA2,
        ka1_mult=KA1_MULT,
        filter_lambda_size=DEFAULT_LAMBDA_SIZE,
        x_mesh=None
        ):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.ka1 = ka1
        self.ka2 = ka2
        self.ka1_mult = ka1_mult
        self.filter_lambda_size = filter_lambda_size
        self.weight = nn.parameter.Parameter(torch.randn(self.filter_lambda_size)*LAMBDA_INIT_WEIGHT_SCALE) # TRASH, just to make the code work
        self.x_mesh = torch.tensor(x_mesh)
        # self.lambda_mesh = torch.linspace(self.lambda1, self.lambda2, filter_lambda_size)

        # Ka2-mesh generation
        self.delta_ka2 = self._delta_twoth(self.ka2, self.x_mesh)
        self.doublet_mesh = self.x_mesh*self.x_mesh/(self.x_mesh + self.delta_ka2)

        self.indices = []
        for i in range(len(self.x_mesh)):
            ge = torch.ge(self.x_mesh[i], self.doublet_mesh)
            s = torch.sum(ge)
            self.indices.append(s)

        self.indices = torch.tensor(self.indices)
        self.indices = torch.clamp(self.indices, 0, len(self.doublet_mesh[1:]) - 1)


    def _torch_interp(self, x, xp, fp):
        """
        Perform 1-D linear interpolation.

        Parameters:
        - x: A 1-D tensor of x-coordinates where to interpolate.
        - xp: A 1-D tensor of x-coordinates of the data points, must be increasing.
        - fp: A 1-D tensor of the same length as xp giving the values of the data points.

        Returns:
        - A tensor of the same shape as x containing the interpolated values.
        """

        # Ensure input tensors are float for interpolation
        x = x.float()
        xp = xp.float()
        fp = fp.float()

        # Indices of the rightmost xp lower than or equal to x
        idxs = torch.searchsorted(xp, x) - 1
        idxs = idxs.clamp(min=0, max=len(xp) - 2)  # Bound the indices to prevent out-of-bounds access

        # Slopes for linear interpolation
        print(fp.shape, xp.shape)
        slopes = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])

        # Calculate interpolated values
        return fp[idxs] + slopes[idxs] * (x - xp[idxs])

    def _delta_twoth(self, lambda_, twoth):
        delta_twoth = 2*torch.rad2deg(
            torch.arcsin(
                lambda_ * torch.sin(torch.deg2rad(0.5*twoth)) / self.ka1
                )
        ) - twoth
        return delta_twoth

    def forward(self, y):
        
        # Ka1 line introduction
        prof = torch.zeros(y.shape)
        prof += y * self.ka1_mult

        # Ka2 line introduction
        doublet_int = self._torch_interp(self.doublet_mesh, self.x_mesh, y[0])
        prof += doublet_int * (1.0 - self.ka1_mult)

        # Bremsstrahlung X-ray introduction
        pass

        return prof


class XRDConv(nn.Module):

    def __init__(self, kernel_size=DEFAULT_FILTER_SIZE, subkernel_size=DEFAULT_SUBFILTER_SIZE, x_data=None, doublet_enabled=True):
        super().__init__()
        self.doublet_enabled = doublet_enabled

        if self.doublet_enabled:
            self.filter_lambda = LambdaConv(
                lambda1=DEFAULT_LAMBDA1,                    # TO BE REWRITTEN
                lambda2=DEFAULT_LAMBDA2,                   # TO BE REWRITTEN
                ka1=MOKA1,                                  # TO BE REWRITTEN
                ka2=MOKA2,                                  # TO BE REWRITTEN
                ka1_mult=KA1_MULT,
                filter_lambda_size=DEFAULT_LAMBDA_SIZE,     # TO BE REWRITTEN
                x_mesh=x_data
            )
        else:
            self.filter_lambda = nn.Module()
            self.filter_lambda.weight = nn.parameter.Parameter(torch.randn(DEFAULT_LAMBDA_SIZE)*LAMBDA_INIT_WEIGHT_SCALE) # TRASH, just to make the code work

        self.filter_instr = InterpolatedConv(
            in_weight_size=subkernel_size,
            out_weight_size=kernel_size
        )

    def forward(self, data):
        if self.doublet_enabled:
            data_lambda = self.filter_lambda(data)
            data_broad = self.filter_instr(data_lambda)
        else:
            data_broad = self.filter_instr(data)
            
        return data_broad


# Loss implementation

class MSEPenalized(nn.Module):

    def __init__(self, L1=DEFAULT_L1):
        super().__init__()
        self.L1 = L1

    def forward(self, input, target, weights):

        input = input.view(-1)
        target = target.view(-1)
        weights = weights.view(-1)

        mse = ((input - target)**2).mean()
        penalty_l1 = self.L1 * ((weights[:-2] + weights[2:] - 2*weights[1:-1])**2).mean()   # TO BE WRITTEN ADEQUATELY
        loss_total = mse + penalty_l1

        return loss_total


class MSEWeighed(nn.Module):

    def __init__(self, inst_factor):
        super().__init__()
        self.inst_factor = inst_factor

    def forward(self, input, target, sigma):
        input = input.view(-1)
        target = target.view(-1)
        sigma = sigma.view(-1)

        mse_w = ((input-target)**2 / (sigma**2 + self.inst_factor)).mean()
        return mse_w


class MEMBurgLoss(nn.Module):

    def __init__(self, inst_factor=DEFAULT_INST_FACTOR):
        super().__init__()
        self.inst_factor = inst_factor

    def _mse_weighed(self, input, target, sigma):
        mse_w = 0.5 * ((input-target)**2 / (sigma**2 + self.inst_factor)).mean()
        return mse_w

    def _burg_entropy(self, f, m):
        burg_ent = torch.mean(
            torch.sqrt(f**2 + m**2)/m - torch.log(torch.sqrt(f**2 + m**2) / (2*m) + 0.5) - 1.0
        )
        return burg_ent

    def forward(self, input, target, sigma, f, m, l1):
        mse_w = self._mse_weighed(input, target, sigma)
        burg_ent = self._burg_entropy(f, m)
        loss = burg_ent + mse_w * l1
        return loss


class MEMLoss(nn.Module):

    def __init__(self, inst_factor=DEFAULT_INST_FACTOR):
        super().__init__()
        self.inst_factor = inst_factor

    def _mse_weighed(self, input, target, sigma):
        mse_w = ((input-target)**2 / (sigma**2 + self.inst_factor)).mean()
        return mse_w

    def _entropy(self, f, m):
        f_norm = f / f.sum()
        m_norm = m / m.sum()
        ent = (f_norm * torch.log(
            (torch.sqrt(f_norm**2 + 4*m_norm**2) + f_norm)/2.0/m_norm
        ) + 2*m_norm - torch.sqrt(f_norm**2 + 4*m_norm**2)).sum()
        return ent, f_norm

    def forward(self, input, target, sigma, f, m, l1, l2):
        mse_w = self._mse_weighed(input, target, sigma)
        ent, f_norm = self._entropy(f, m)
        loss = ent + (mse_w - 1.0) * l1 + (f_norm.sum() - 1.0) * l2
        return loss


class RLLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.view(-1)
        target = target.view(-1)

        alpha = (target*torch.log(input) - input).sum()
        return alpha


# Data preprocessor implementation

class Preprocessor():

    def __init__(
        self,
        lambda1=CUKA1,
        lambda2=CUKA2,
        twotheta_step=DEFAULT_TWOTHETA_STEP
        ):
        self.lambda1, self.lambda2 = lambda1, lambda2
        self.twotheta_step = twotheta_step

    def _delta_twoth(self, twoth):
        delta_twoth = 2*np.rad2deg(
            np.arcsin(
                self.lambda2 * np.sin(np.deg2rad(0.5*twoth)) / self.lambda1
                )
        ) - twoth
        return delta_twoth

    def doublet_fast(self, data):
        prof = np.zeros(data.shape)
        mesh = data[:, 0]
        delta = self._delta_twoth(mesh)
        douplet_mesh = mesh*mesh/(mesh+delta)
        duplet = np.interp(douplet_mesh, mesh, data[:, 1])
        print(mesh.shape, data[:, 1].shape)
        prof[:, 0] = data[:, 0]
        prof[:, 1] = KA1_MULT * data[:, 1] + (1.0 - KA1_MULT) * duplet
        return prof

    def norm_max(self, data):
        data_norm = data / np.max(data)
        return data_norm

    def norm_int(self, data, target):
        data_norm = data * target.sum() / data.sum()
        return data_norm


# Custom Dataset implementation

class CustomDataset(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        label_idx = self.label[idx]
        data_idx = self.data[idx]

        if self.transform:
            data_idx = self.transform(data_idx)

        return data_idx.float(), label_idx.float()