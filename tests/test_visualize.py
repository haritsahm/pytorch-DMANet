import cv2
import numpy as np
import pytest

import src.utils.visualize as viz_tool
from src.datamodules.dmanet_datamodule import DMANetDataModule


@pytest.fixture
def config():
    return {
        'data_dir': 'data/cityscape_fo_segmentation',
        'dataloader': 'src.datamodules.components.fiftyone_dataset.ImageSegmentationDirectory',
        'num_classes': 19,
        'image_size': [640, 640],
        'train_transform': None,
        'test_transform': None,
        'batch_size': 4,
        'num_workers': 0,
        'pin_memory': False,
    }


def test_get_color_mask(config):

    datamodule = DMANetDataModule(**config)
    datamodule.setup()
    images, targets = next(iter(datamodule.train_dataloader()))

    images = images.cpu().numpy()
    targets = targets.cpu().numpy()

    for idx, (image, target) in enumerate(zip(images, targets)):
        image = (image * 255).astype(np.uint8)
        color_img = viz_tool.show_prediction(image.transpose((1, 2, 0)), target, overlay=0.5)
        cv2.imwrite(f'sample-{idx}.jpg', color_img)
