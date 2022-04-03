from typing import List

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import nn

from src.models.functions import layers, weight_init


class MultiAggregationNetwork(nn.Module):
    # TODO: Docstring

    def __init__(self, num_classes: int, channels: List, input_size: List):
        super().__init__()

        base, low, mid, high = channels
        self._input_size = np.array(input_size)

        # LERB layers
        self._low_lerb = layers.LatticeEnhancedBlock(x_channels=low // 4, m_channels=base)
        self._mid_lerb = layers.LatticeEnhancedBlock(x_channels=mid // 2, m_channels=base)
        self._high_lerb = layers.LatticeEnhancedBlock(x_channels=high // 4, m_channels=base)

        # Downsampling CBR layers
        self._low_cbr = nn.Sequential(
            layers.ConvBNReLU(in_channels=low, out_channels=low // 2),
            layers.ConvBNReLU(in_channels=low // 2, out_channels=low // 4),
        )

        self._mid_cbr = nn.Sequential(
            layers.ConvBNReLU(in_channels=mid, out_channels=mid // 2),
            layers.ConvBNReLU(in_channels=mid // 2, out_channels=mid // 2),
        )

        self._high_cbr = nn.Sequential(
            layers.ConvBNReLU(in_channels=high, out_channels=high // 2),
            layers.ConvBNReLU(in_channels=high // 2, out_channels=high // 4),
        )

        # GCB Layer
        self._gcb = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            layers.ConvBNReLU(in_channels=high, out_channels=high // 2),
        )

        # FTB layers
        self._high_ftb = layers.FeatureTransformationBlock(in_channels=high // 2)
        self._mid_ftb = layers.FeatureTransformationBlock(in_channels=low // 2)

        # Upsampling CBR
        self._upmid_cbr = layers.ConvBNReLU(
            in_channels=mid, out_channels=low // 2)
        self._uplow_cbr = layers.ConvBNReLU(
            in_channels=low // 2, out_channels=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.kaiming_init(m)

    def forward(self, x: List) -> torch.Tensor:
        # TODO: Docstring

        c2, c3, c4, c5 = x

        low_features = self._low_lerb(
            self._low_cbr(c3), F.max_pool2d(c2, kernel_size=2, stride=2))
        mid_features = self._mid_lerb(
            self._mid_cbr(c4), F.max_pool2d(c2, kernel_size=3, stride=4))
        high_features = self._high_lerb(
            self._high_cbr(c5), F.max_pool2d(c2, kernel_size=7, stride=8))
        gcb_features = self._gcb(c5)

        features = F.interpolate(gcb_features, size=tuple(
            self._input_size // 32), mode='bilinear', align_corners=True)
        features = features + high_features
        features = self._high_ftb(features)
        high_aux = features

        features = F.interpolate(features, size=tuple(
            self._input_size // 16), mode='bilinear', align_corners=True)
        features = features + mid_features
        features = self._upmid_cbr(features)
        features = self._mid_ftb(features)
        mid_aux = features

        features = F.interpolate(features, size=tuple(
            self._input_size // 8), mode='bilinear', align_corners=True)
        features = features + low_features
        features = self._uplow_cbr(features)

        features = F.interpolate(features, size=tuple(self._input_size),
                                 mode='bilinear', align_corners=True)

        if self.training:
            return features, mid_aux, high_aux
        else:
            return features


class DMANet(nn.Module):
    # TODO: Docstring

    def __init__(
        self,
        num_classes: int = 19,
        input_size: List = [640, 640],
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
            channels=self._encoder.feature_info.channels(),
            input_size=input_size)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def input_size(self):
        return self._input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Docstring

        enc_features = self._encoder(x)
        dec_masks = self._decoder(enc_features)

        return dec_masks
