import glob
import os
import pathlib
from typing import Optional

import albumentations as Albu
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

CLASS_NAMES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', 'traffic_sign',
               'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
IGNORE_INDEX = 255
VALID_CLASSES = [7, 8, 11, 12, 13, 17, 19,
                 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
VOID_CLASSES = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
CLASS_MAP = dict(zip(VALID_CLASSES, range(19)))


class ImageSegmentationDirectory(Dataset):
    # TODO: Docstring

    def __init__(self, dataset_dir: str, stage: str = 'train', transform: Optional[Albu.Compose] = None):
        dataset_dir = pathlib.Path(dataset_dir)
        self._dataset_dir = str(dataset_dir)

        self._labels = sorted(glob.glob(str(dataset_dir / f'labels_{stage}' / '*.*')))
        self._images = sorted(glob.glob(
            str(dataset_dir / f'data_{stage}' / '*.*'))) if (dataset_dir / f'data_{stage}').exists() else []

        self._stage = stage
        if stage != 'test':
            self._images = sorted(
                [x for x in self._images if x.replace(f'data_{stage}', f'labels_{stage}') in self._labels])

        self._transform = transform

    def __len__(self):
        # TODO Docstirng

        return len(self._images)

    def __getitem__(self, idx):
        # TODO: Docstring

        img_path = self._images[idx]
        mask_path = self._labels[idx] if self._stage != 'test' else None

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.ones(image.shape[:2] + (1,), dtype=np.uint8) * 255
        mask = cv2.imread(mask_path).astype(np.uint8) if self._stage != 'test' else mask

        if self._transform is not None:
            transformed = self._transform(image=image, mask=mask)

            image = transformed['image']
            mask = transformed['mask']

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        return image, mask

    def collate_fn(self, batch):
        pass
