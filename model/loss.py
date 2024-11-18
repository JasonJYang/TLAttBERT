import torch
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

def crossentropy_loss(output, target):
    return F.cross_entropy(output, target)

def rmse_loss(output, target):
    return torch.sqrt(F.mse_loss(output, target))