from typing import Union

import torch
import torch.nn.functional as F
from torch import nn


class FeatureTransformationBlock(nn.Module):
    """Feature Transformation Block Module.

    FTB is comprised of two main sub-branches to perform attention
    operations along the channel and spatial dimensions.
    A weight learning sub-branch is used to adaptively learn the
    weights for the channel sub-branch and the spatial sub-branch

    Source: https://arxiv.org/pdf/2203.04037.pdf Section III. C.2

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    """

    def __init__(self, in_channels: int):
        super().__init__()

        self._cbr = ConvBNReLU(in_channels=in_channels, out_channels=in_channels)

        # Weight learning sub-branch
        self._wlb = nn.Sequential(
            nn.Linear(1, 1),
            nn.Softmax2d(),
        )

        # Channel sub-branch
        self._csb = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(1, 1),
        )

        # Spatial sub-branch
        self._ssb = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = self._cbr(x)
        x_g = F.adaptive_avg_pool2d(x_f, output_size=(1, 1))

        v, w = [self._wlb(x_g) for _ in range(2)]

        x_s = self._ssb(x_f)
        x_s = x_s.expand(-1, v.shape[1], *x_s.shape[2:]) * v.expand(-1, v.shape[1], *x_s.shape[2:])
        x_c = self._csb(x_g)
        x_c = x_c.expand(-1, w.shape[1], *x_c.shape[2:]) * w.expand(-1, w.shape[1], *x_c.shape[2:])

        t = F.sigmoid(x_s + x_c.expand(x_s.shape))
        return t * x_f


class LatticeEnhancedBlock(nn.Module):
    """Lattice Enhanced Residual Block Module.

    Implement the LERB module cosist of a Contextual Module and a Spatial Module.
    Each module consists of an enhanced sub-module, a spatial Weight Learning Block (WLB),
    and a Lattice Structure.

    Source: https://arxiv.org/pdf/2203.04037.pdf Section III. C.1

    Parameters
    ----------
    x_channels : int
        Number of input channel from sub branch.
    m_channels : int
        Number of input channel form downsampled feature map.
    """

    def __init__(self, x_channels: int, m_channels: int):
        super().__init__()

        # Contextual Enhanced Block
        self._ce_block = nn.Sequential(
            nn.Conv2d(x_channels, x_channels, kernel_size=3, padding=2, stride=1, dilation=2),
            nn.BatchNorm2d(x_channels),
            nn.Conv2d(x_channels, x_channels, kernel_size=3, padding=4, stride=1, dilation=4),
            nn.BatchNorm2d(x_channels),
        )

        # Spatial Enhanced Block
        self._se_block = nn.Sequential(
            nn.Conv2d(x_channels + m_channels, x_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(x_channels),
        )

        # CM - Weight Learning Block
        self._cm_wlb = nn.Sequential(
            nn.Conv2d(in_channels=x_channels, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        # SM - Weight Learning Block
        self._sm_wlb = nn.Sequential(
            nn.Conv2d(in_channels=x_channels, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward_spatial(self, x: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Compute the spatial sub-module."""

        a_s, b_s = [self._sm_wlb(x) for _ in range(2)]
        se_x = self._se_block(torch.cat((x, M), 1))

        p = F.relu(b_s.expand(x.shape) * x + se_x)
        q = F.relu(x + a_s.expand(se_x.shape) * se_x)

        return p + q

    def forward_contextual(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the contextual sub-module."""

        a_c, b_c = [self._cm_wlb(x) for _ in range(2)]
        ce_x = self._ce_block(x)

        p = F.relu(x + b_c.expand(ce_x.shape) * ce_x)
        q = F.relu(a_c.expand(x.shape) * x + ce_x)

        return p + q

    def forward(self, X: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        f_c = self.forward_contextual(X)
        f_s = self.forward_spatial(f_c, M)

        return torch.cat((f_c, f_s), 1)


class ConvBNReLU(nn.Module):
    """Convolution-Batchnorm-ReLU Block.

    Generate a standard Conv-Bn-RelU (CBR) layer

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int, optional
        Kernel size, by default 3
    stride : int, optional
        Stride size, by default 1
    padding : Union[int, str], optional
        Padding size, by default 1
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: Union[int, str] = 1,
                 use_activation=True):

        super(ConvBNReLU, self).__init__()
        self._conv = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=False)
        self._bn = nn.BatchNorm2d(out_channels)
        self._use_activation = use_activation
        if use_activation:
            self._relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self._conv(x)
        x = self._bn(x)
        if self._use_activation:
            x = self._relu(x)

        return x
