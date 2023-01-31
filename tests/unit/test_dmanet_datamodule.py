import os
import tempfile

import albumentations as Albu
import hydra
import omegaconf as oc
import pytest
import torch
from hydra import compose, initialize, initialize_config_module
from torch.utils.data import Dataset
from torchvision import transforms

from src.datamodules.dmanet_datamodule import DMANetDataModule


def configs():
    return [{
        'data_dir': '/media/haritsahm/DataStorage/dataset/cityscapes/cityscape_fo_segmentation',
        'dataloader': 'src.datamodules.components.fiftyone_dataset.ImageSegmentationDirectory',
        'num_classes': 19,
        'image_size': [768, 1536],
        'train_transform': None,
        'test_transform': None,
        'batch_size': 4,
        'num_workers': 0,
        'pin_memory': False,
    }]


@pytest.mark.parametrize('config', configs())
def test_dmanet_datamodule(config):
    print(config)
    datamodule = DMANetDataModule(**config)

    assert not datamodule.data_train and not datamodule.data_val and not datamodule.data_test

    assert os.path.exists(config['data_dir'])

    datamodule.setup()

    assert isinstance(datamodule.data_train, Dataset) and isinstance(
        datamodule.data_val, Dataset) and isinstance(datamodule.data_test, Dataset)
    assert (
        len(datamodule.data_train) + len(datamodule.data_val) + len(datamodule.data_test) == 5000
    )

    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()
    assert train_dataloader
    assert val_dataloader
    assert test_dataloader

    batch = next(iter(train_dataloader))
    x, y = batch

    assert len(x) == config['batch_size']
    assert len(y) == config['batch_size']
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


@pytest.mark.parametrize('config', configs())
def test_dmanet_transformations(config):

    train_transform = Albu.Compose([
        Albu.OneOf([
            Albu.RandomSizedCrop(min_max_height=(50, 101), height=640,
                                 width=640, p=0.5),
            Albu.PadIfNeeded(min_height=640, min_width=640, p=0.5)
        ], p=1),
        Albu.VerticalFlip(p=0.5),
        Albu.RandomRotate90(p=0.5),
        Albu.OneOf([
            Albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            Albu.GridDistortion(p=0.5),
            Albu.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)
        ], p=0.8),
        Albu.CLAHE(p=0.8),
        Albu.RandomBrightnessContrast(p=0.8),
        Albu.RandomGamma(p=0.8)
    ])

    config['train_transform'] = train_transform

    datamodule = DMANetDataModule(**config)

    assert isinstance(datamodule._train_transform, Albu.Compose)
    assert datamodule._train_transform == train_transform

    temp = tempfile.NamedTemporaryFile()

    Albu.save(train_transform, temp.name, data_format='yaml')

    config['train_transform'] = hydra.utils.instantiate({
        '_target_': 'albumentations.core.serialization.load',
        'filepath': temp.name,
        'data_format': 'yaml',
    })

    datamodule = DMANetDataModule(**config)
    assert isinstance(datamodule._train_transform, Albu.Compose)

    datamodule.setup()
    assert datamodule.train_dataloader()

    train_transform = transforms.Compose([transforms.CenterCrop((178, 178)),
                                          transforms.Resize(128),
                                          transforms.RandomRotation(20),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    config['train_transform'] = train_transform

    with pytest.raises(TypeError):

        datamodule = DMANetDataModule(**config)
