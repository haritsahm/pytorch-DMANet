# Source: https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/model/utils/weight_init.py

import numpy as np
import torch.nn as nn


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    """Initialize module weight using Xavier method."""

    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    """Initialize module weight using Normal Distribution method."""

    nn.init.normal_(module.weight, mean, std)
    if module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(module, a=0, b=1, bias=0):
    """Initialize module weight using Uniform Distribution method."""

    nn.init.uniform_(module.weight, a, b)
    if module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    """Initialize module weight using Kaiming method."""

    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, mode=mode, nonlinearity=nonlinearity)
    if module.bias is not None:
        nn.init.constant_(module.bias, bias)


def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to giving probablity."""

    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init
