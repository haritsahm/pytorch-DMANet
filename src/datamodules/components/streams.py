from typing import Optional, Tuple

import albumentations as Albu
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.io import VideoReader as TorchVideoReader


class VideoReader(Dataset):
    def __init__(self, dataset_dir: str, stage: str = 'test', transform: Optional[Albu.Compose] = None):
        self._dataset_dir = dataset_dir
        self._video = cv2.VideoCapture(self._dataset_dir)
        if not self._video.isOpened():
            raise RuntimeError("Video file not valid")
        self._transform = transform
        self._output_videos = True

    def __len__(self) -> int:
        return int(self._video.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def video_data(self):
        return True

    @property
    def dataset_dir(self):
        return self._dataset_dir

    @property
    def video_fps(self):
        return int(self._video.get(cv2.CAP_PROP_FPS))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        _, image = self._video.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.zeros(image.shape[:2]).astype(np.uint8)

        if self._transform is not None:
            transformed = self._transform(image=image, mask=mask)

            image = transformed['image']
            mask = transformed['mask']

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
            image /= 255.0

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        mask = mask.to(torch.long)

        return image, mask
