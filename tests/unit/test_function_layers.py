import pytest
import torch

import src.models.functions.layers as layers


@pytest.mark.parametrize('batch_size,input_shape,in_channels',
                         [(2, [48, 96], 1280), (4, [24, 48], 128),
                          (8, [20, 20], 1280), (16, [40, 40], 128)])
def test_ftb_layer(batch_size, input_shape, in_channels):

    x = torch.rand((batch_size, in_channels, *input_shape))
    layer = layers.FeatureTransformationBlock(in_channels)

    out = layer(x)

    assert out.shape == x.shape


@pytest.mark.parametrize('batch_size, x_input_shape,m_input_shape',
                         [(2, [32, 96, 192], [64, 96, 192]),
                          (4, [64, 48, 96], [64, 48, 96]),
                          (8, [128, 24, 48], [64, 24, 48]),
                          (2, [32, 80, 80], [64, 80, 80]),
                          (4, [64, 40, 40], [64, 40, 40]),
                          (8, [128, 20, 20], [64, 20, 20])])
def test_lteb_layer(batch_size, x_input_shape, m_input_shape):

    x = torch.rand(batch_size, *x_input_shape)
    m = torch.rand(batch_size, *m_input_shape)

    layer = layers.LatticeEnhancedBlock(x_channels=x.shape[1], m_channels=m.shape[1])

    out = layer(x, m)

    assert out.shape == (x.shape[0], 2 * x.shape[1], *x.shape[2:])
