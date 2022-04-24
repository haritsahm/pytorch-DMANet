import time

import numpy as np
import pytest
import torch
import torchinfo

import src.models.components.dma_net as net


@pytest.mark.parametrize('batch_size,input_shape',
                         [(2, [192, 384]), (4, [192, 384]), (8, [192, 384]),
                          (2, [160, 160]), (4, [160, 160]), (8, [160, 160])])
def test_man_layer(batch_size, input_shape):
    input_shape = np.array(input_shape)
    c2 = torch.rand(batch_size, 64, *input_shape)
    c3 = torch.rand(batch_size, 128, *(input_shape // 2))
    c4 = torch.rand(batch_size, 256, *(input_shape // 4))
    c5 = torch.rand(batch_size, 512, *(input_shape // 8))

    layer = net.MultiAggregationNetwork(
        num_classes=19, channels=[64, 128, 256, 512], input_size=input_shape * 4)

    features, mid_aux, high_aux = layer([c2, c3, c4, c5])

    assert features.shape == (batch_size, 19, *(input_shape * 4))
    assert mid_aux.shape == (batch_size, 64, *(input_shape // 4))
    assert high_aux.shape == (batch_size, 256, *(input_shape // 8))


@pytest.mark.parametrize('batch_size,input_shape',
                         [(2, [640, 640]), (4, [640, 640]), (6, [640, 640]),
                          (2, [768, 1536]), (4, [768, 1536]), (6, [768, 1536])])
def test_dmanet(batch_size, input_shape):

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    input_shape = np.array(input_shape)
    x = torch.rand(batch_size, 3, *input_shape.tolist(), device=device)
    model = net.DMANet(num_classes=19, input_size=input_shape.tolist())
    model.to(device)

    # Model for training
    model.train()
    output, mid_aux, high_aux = model(x)

    # Check training output
    assert output.shape == (batch_size, 19, *input_shape.tolist())
    assert mid_aux.shape == (batch_size, 64, *(input_shape // 16).tolist())
    assert high_aux.shape == (batch_size, 256, *(input_shape // 32).tolist())

    # Model for inference
    model.eval()
    x = torch.rand(batch_size // 2, 3, *input_shape.tolist(), device=device)
    with torch.no_grad():
        t = time.time()
        output = model(x)
        infer_t = time.time() - t

    # Check inference output
    assert output.shape == (batch_size // 2, 19, *input_shape.tolist())

    # Check FPS
    assert 1 / infer_t > 40

    summary = torchinfo.summary(model, x.shape, verbose=0, col_names=[
                                'input_size', 'output_size', 'num_params'])

    # Check Paramsize
    assert summary.total_params < 20e+6
