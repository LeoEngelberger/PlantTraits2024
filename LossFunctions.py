import os
os.environ["KERAS_BACKEND"] = "torch"
import keras as ks
from torch import nn


import torch
import torch.nn as nn
import torch.nn.functional as F

class R2Loss(nn.Module):
    def __init__(self, use_mask=False):
        super(R2Loss, self).__init__()
        self.use_mask = use_mask

    def forward(self, y_true, y_pred):
        if self.use_mask:
            mask = (y_true != -1)
            y_true = torch.where(mask, y_true, torch.tensor(0.0))
            y_pred = torch.where(mask, y_pred, torch.tensor(0.0))
        SS_residue = torch.sum(torch.square(y_true - y_pred))
        SS_total = torch.sum(torch.square(y_true - torch.mean(y_true)))
        r2_loss = SS_residue / (SS_total + 1e-6)
        return torch.mean(r2_loss)


class R2Metric(nn.Module):
    def __init__(self):
        super(R2Metric, self).__init__()
        self.register_buffer('SS_residual', torch.zeros(6))
        self.register_buffer('SS_total', torch.zeros(6))
        self.register_buffer('num_samples', torch.tensor(0.0))

    def update_state(self, y_true, y_pred, sample_weight=None):
        SS_residual = torch.sum(torch.square(y_true - y_pred))
        SS_total = torch.sum(torch.square(y_true - y_pred))
        self.SS_residual.add_(SS_residual)
        self.SS_total.add_(SS_total)
        self.num_samples.add_(y_true.size(0))

    def result(self):
        r2 = 1 - self.SS_residual / (self.SS_total + 1e-6)
        return torch.mean(r2)

    def reset_states(self):
        self.SS_residual.zero_()
        self.SS_total.zero_()
        self.num_samples.zero_()