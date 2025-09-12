import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d
import torch
import torch.nn.functional as F
from torch import nn


class weighted_l1_loss(nn.Module):
    def __init__(self):
        super(weighted_l1_loss, self).__init__()

    def forward(self, inputs, targets, all_weights):
        targets_list = targets.tolist()
        weights = []
        for target in targets_list:
            weights.append(all_weights[round(target[0])])
        weights = torch.tensor(weights, device=inputs.device).unsqueeze(dim=1)
        loss = F.smooth_l1_loss(inputs, targets, reduction='none')
        if weights is not None:
            loss *= weights.expand_as(loss)
        loss = torch.mean(loss)
        return loss


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(
            map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window
