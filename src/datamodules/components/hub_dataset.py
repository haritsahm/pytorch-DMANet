from typing import Optional, Tuple

import albumentations as Albu
import hub
import numpy as np
import torch
from torch.utils.data import Dataset


class HubSegmentationDataset(Dataset):

    def __init__(self, dataset_dir: str, stage: str = 'train', transform: Optional[Albu.Compose] = None):
        for split in ['train', 'val', 'test']:
            if split in dataset_dir:
                if 'validation' in dataset_dir:
                    dataset_dir = dataset_dir.replace('validation', 'val')
                elif 'val' in dataset_dir:
                    pass
                else:
                    dataset_dir = dataset_dir.replace(split, stage)
        self._ds = hub.load(dataset_dir, read_only=True)
        self._transform = transform

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        image = self._ds.images[idx].numpy()
        mask = self._ds.segmentations[idx].numpy(fetch_chunks=True).astype(np.uint8)

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
