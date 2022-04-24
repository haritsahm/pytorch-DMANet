import glob
import os
import pathlib
from typing import Optional, Tuple

import albumentations as Albu
import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset

CLASS_NAMES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', 'traffic_sign',
               'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus',
               'train', 'motorcycle', 'bicycle']
IGNORE_INDEX = 255
VALID_CLASSES = [7, 8, 11, 12, 13, 17, 19,
                 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
VOID_CLASSES = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
CLASS_MAP = dict(zip(VALID_CLASSES, range(19)))


class ImageSegmentationDirectory(Dataset):
    """ImageSegmentation Dataset format reader.

    For full documentation of ImageSegmentation please read the docs.
    https://voxel51.com/docs/fiftyone/user_guide/export_datasets.html#imagesegmentationdirectory

    Parameters
    ----------
    dataset_dir : str
        Dataset root directory
    stage : str, optional
        Dataset stage option, by default 'train'
    transform : Optional[Albu.Compose], optional
        Data transformation pipeline, by default None
    """

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
        return len(self._images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Load dataset item.

        Parameters
        ----------
        idx : int
            Data index

        Returns
        -------
        Tuple[torch.Tensor, torch.LongTensor]
            Tuples of image input and target segmentation mask
        """

        img_path = self._images[idx]
        mask_path = self._labels[idx] if self._stage != 'test' else None

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.ones(image.shape[:2] + (1,), dtype=np.uint8) * 255
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(
            np.uint8) if self._stage != 'test' else mask

        if mask.ndim == 3:
            mask = np.squeeze(mask)

        if self._transform is not None:
            transformed = self._transform(image=image, mask=mask)

            image = transformed['image']
            mask = transformed['mask']

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        mask = mask.to(torch.long)

        return image, mask

    def collate_fn(self, batch):
        pass


class COCOSegmentation(Dataset):
    """COCO Dataset format reader.

    For full documentation of COCO dataset please read the docs.
    https://voxel51.com/docs/fiftyone/user_guide/export_datasets.html#cocodetectiondataset

    Parameters
    ----------
    dataset_dir : str
        Dataset root directory
    stage : str, optional
        Dataset stage option, by default 'train'
    transform : Optional[Albu.Compose], optional
        Data transformation pipeline, by default None
    """

    def __init__(
        self,
            dataset_dir: str,
            stage: str = 'train',
            transform: Optional[Albu.Compose] = None) -> None:
        super().__init__()
        img_dir = os.path.join(dataset_dir, 'data')
        annotation_path = os.path.join(dataset_dir, 'labels', f'{stage}.json')
        self._ann = COCO(annotation_path)
        self._img_data = self._ann.loadImgs(ids=self._ann.getImgIds())
        self._catIds = self._ann.getCatIds()
        self._file_paths = [str(pathlib.Path(img_dir) / img['file_name']) for img in self._img_data]
        self._transform = transform
        self._img_dir = img_dir

    def __len__(self) -> int:
        return len(self._file_paths)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Load dataset item.

        Parameters
        ----------
        idx : int
            Data index

        Returns
        -------
        Tuple[torch.Tensor, torch.LongTensor]
            Tuples of image input and target segmentation mask
        """

        ann_ids = self._ann.getAnnIds(
            imgIds=self._img_data[i]['id'],
            catIds=self._catIds,
            iscrowd=None
        )
        anns = self._ann.loadAnns(ann_ids)

        image = cv2.imread(self._file_paths[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = np.max(np.stack([self._ann.annToMask(ann) * ann['category_id']
                                for ann in anns]), axis=0)

        if self._transform is not None:
            transformed = self._transform(image=image, mask=mask)

            image = transformed['image']
            mask = transformed['mask']

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        mask = mask.to(torch.long)

        return image, mask
