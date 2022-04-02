from typing import Optional, Tuple

import albumentations as Albu
import hydra
import omegaconf as oc
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class CityscapeDataModule(LightningDataModule):
    """Cityscape Wrapped in Lightning Datamodule.

    # TODO: Docstring

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    """

    def __init__(
        self,
        data_dir: str,
        dataloader: str,
        num_classes: int = 19,
        train_transform: Optional[Albu.Compose] = None,
        test_transform: Optional[Albu.Compose] = None,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        if train_transform and not isinstance(train_transform, Albu.Compose):
            raise TypeError('Train transform is not an Albumentation Compose')
        if test_transform and not isinstance(test_transform, Albu.Compose):
            raise TypeError('Test transform is not an Albumentation Compose')

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations
        self._train_transform: Albu.Compose = Albu.Compose([
            Albu.RandomCrop(width=640, height=640),
            Albu.HorizontalFlip(p=0.5),
            Albu.RandomBrightnessContrast(p=0.2),
            Albu.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            ToTensorV2(),
        ])

        self._test_transform: Albu.Compose = Albu.Compose([
            Albu.Resize(width=640, height=640),
            Albu.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            ToTensorV2(),
        ])

        self._train_transform = train_transform if isinstance(
            train_transform, Albu.Compose) else self._train_transform
        self._test_transform = test_transform if isinstance(
            test_transform, Albu.Compose) else self._test_transform

        # Dataloader class
        self._loader_cls = {'dataloader': {'_target_': dataloader}}

        self._data_train: Optional[Dataset] = None
        self._data_val: Optional[Dataset] = None
        self._data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        # TODO: Docstring

        return self.hparams.num_classes

    @property
    def data_train(self):
        # TODO: Docstring

        return self._data_train

    @property
    def data_val(self):
        # TODO: Docstring

        return self._data_val

    @property
    def data_test(self):
        # TODO: Docstring

        return self._data_test

    def setup(self):
        """Load data.

        Set variables: `self._data_train`, `self._data_val`, `self._data_test`.
        """

        # load datasets only if they're not loaded already
        if not self._data_train and not self._data_val and not self._data_test:
            self._data_train = hydra.utils.instantiate(
                self._loader_cls['dataloader'], dataset_dir=self.hparams.data_dir, stage='train', transform=self._train_transform)
            self._data_val = hydra.utils.instantiate(
                self._loader_cls['dataloader'], dataset_dir=self.hparams.data_dir, stage='validation', transform=self._test_transform)
            self._data_test = hydra.utils.instantiate(
                self._loader_cls['dataloader'], dataset_dir=self.hparams.data_dir, stage='test', transform=self._test_transform)

    def train_dataloader(self):
        # TODO: Docstring

        return DataLoader(
            dataset=self._data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        # TODO: Docstring

        return DataLoader(
            dataset=self._data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        # TODO: Docstring

        return DataLoader(
            dataset=self._data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
