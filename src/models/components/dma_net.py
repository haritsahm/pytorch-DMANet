from typing import Iterable, List

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import nn

from src.models.functions import layers, weight_init


class MultiAggregationNetwork(nn.Module):
    """Multi Aggregation Netowrk Module.

    Source: https://arxiv.org/pdf/2203.04037.pdf Section III. C

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    backbone_channels_size : Tuple
        Tuple of different levels of sub-network feature maps.
    low_level_features: int, optional
        Feature size for the low resolution block
    high_level_features: int, optional
        Feature size for the high resolution block
    input_size : Tuple
        Tuple of network input sizes.
    """

    def __init__(self,
                 num_classes: int,
                 backbone_channels_size: List[int],
                 low_level_features: int = 128,
                 high_level_features: int = 128,
                 input_size: List[int] = [640, 640]
                 ):
        super().__init__()

        base, low, mid, high = backbone_channels_size
        low_level_features = low_level_features if low_level_features >= low else low
        high_level_features = high_level_features if high_level_features >= low else low
        self._input_size = np.array(input_size)

        # LERB layers
        self._low_lerb = layers.LatticeEnhancedBlock(x_channels=low_level_features//2, m_channels=base)
        self._mid_lerb = layers.LatticeEnhancedBlock(x_channels=high_level_features//2, m_channels=base)
        self._high_lerb = layers.LatticeEnhancedBlock(x_channels=high_level_features//2, m_channels=base)

        # Downsampling CBR layers
        self._low_cbr = nn.Sequential(
            layers.ConvBNReLU(in_channels=low, out_channels=low // 2),
            layers.ConvBNReLU(in_channels=low // 2, out_channels=low_level_features//2),
        )

        self._mid_cbr = nn.Sequential(
            layers.ConvBNReLU(in_channels=mid, out_channels=mid // 2),
            layers.ConvBNReLU(in_channels=mid // 2, out_channels=high_level_features//2),
        )

        self._high_cbr = nn.Sequential(
            layers.ConvBNReLU(in_channels=high, out_channels=high // 2),
            layers.ConvBNReLU(in_channels=high // 2, out_channels=high_level_features//2),
        )

        # GCB Layer
        self._gcb_conv = layers.ConvBNReLU(high, high_level_features, padding='same')

        # FTB layers
        self._high_ftb = layers.FeatureTransformationBlock(in_channels=high_level_features)
        self._mid_ftb = layers.FeatureTransformationBlock(in_channels=low_level_features)

        # Upsampling CBR
        self._upmid_cbr = layers.ConvBNReLU(
            in_channels=high_level_features, out_channels=low_level_features)
        self._uplow_cbr = layers.ConvBNReLU(
            in_channels=low_level_features, out_channels=num_classes, use_activation=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.kaiming_init(m)

    def forward(self, x: Iterable) -> torch.Tensor:
        c2, c3, c4, c5 = x

        low_features = self._low_lerb(
            self._low_cbr(c3), F.max_pool2d(c2, kernel_size=2, stride=2))
        mid_features = self._mid_lerb(
            self._mid_cbr(c4), F.max_pool2d(c2, kernel_size=3, stride=4))
        high_features = self._high_lerb(
            self._high_cbr(c5), F.max_pool2d(c2, kernel_size=7, stride=8))

        gcb_features = torch.mean(c5, dim=(2, 3), keepdim=True)
        gcb_features = self._gcb_conv(gcb_features)

        features = F.interpolate(gcb_features, size=tuple(
            self._input_size // 32), mode='bilinear', align_corners=True)
        features = features + high_features
        features = self._high_ftb(features)
        high_aux = features.clone()

        features = F.interpolate(features, size=tuple(
            self._input_size // 16), mode='bilinear', align_corners=True)
        features = features + mid_features
        features = self._upmid_cbr(features)
        features = self._mid_ftb(features)
        mid_aux = features.clone()

        features = F.interpolate(features, size=tuple(
            self._input_size // 8), mode='bilinear', align_corners=True)
        features = features + low_features
        features = self._uplow_cbr(features)

        features = F.interpolate(features, size=tuple(self._input_size), mode='bilinear', align_corners=True)

        if self.training:
            return features, mid_aux, high_aux
        else:
            return features


class DMANet(nn.Module):
    """DMA-Net Model.

    Parameters
    ----------
    num_classes : int, optional
        Number of output classes, by default 19
    input_size : List, optional
        List of network input sizes, by default [640, 640]
    low_level_features: int, optional
        Feature size for the low resolution block
    high_level_features: int, optional
        Feature size for the high resolution block
    backbone_type : str, optional
        Backbone type for timm model constructor, by default 'resnet18'
    backbone_pretrained : bool, optional
        Use pretrained model, by default True
    """

    def __init__(
        self,
        num_classes: int = 19,
        input_size: List[int] = [640, 640],
        low_level_features: int = 128,
        high_level_features: int = 128,
        backbone_type: str = 'resnet18',
        backbone_pretrained: bool = True,
    ):
        super().__init__()

        self._num_classes = num_classes
        self._input_size = input_size

        self._encoder = timm.create_model(
            backbone_type, pretrained=backbone_pretrained,
            features_only=True, out_indices=(1, 2, 3, 4))
        self._decoder = MultiAggregationNetwork(
            num_classes=num_classes,
            backbone_channels_size=self._encoder.feature_info.channels(),
            low_level_features=low_level_features,
            high_level_features=high_level_features,
            input_size=input_size)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def input_size(self):
        return self._input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_features = self._encoder(x)
        dec_masks = self._decoder(enc_features)

        return dec_masks
