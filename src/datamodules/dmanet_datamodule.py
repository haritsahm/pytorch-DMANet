from typing import List, Optional

import albumentations as Albu
import hydra
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class DMANetDataModule(LightningDataModule):
    """Datamodule Wrapped in Lightning Datamodule.

    Lightning datamodule for DMANet dataset.
    The datamodule can be initialized using different dataset formats depending on the users.
    Albumentation is the default transformation pipeline for easy to use and reconfigurable setups.
    For full documentation of LightningDataModule, plese read the docs.
    Source: https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html

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
        image_size: List = [640, 640],
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

        im_height, im_width = image_size

        # data transformations
        self._train_transform: Albu.Compose = Albu.Compose([
            Albu.HorizontalFlip(p=0.5),
            Albu.RandomScale((-0.5, 1.0), interpolation=2),
            Albu.SmallestMaxSize(max_size=int(im_width*1.5), interpolation=2),
            Albu.RandomCrop(width=im_width, height=im_height),
            Albu.RandomBrightnessContrast(p=0.2),
            Albu.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            ToTensorV2(),
        ])

        self._test_transform: Albu.Compose = Albu.Compose([
            Albu.Resize(width=im_width, height=im_height),
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
        self._data_predict: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        """Get number of classes."""

        return self.hparams.num_classes

    @property
    def data_train(self):
        """Train dataset."""
        return self._data_train

    @property
    def data_val(self):
        """Validation dataset."""
        return self._data_val

    @property
    def data_test(self):
        """Test dataset."""
        return self._data_test

    @property
    def data_predict(self):
        """Prediction dataset."""
        return self._data_predict

    def setup(self, stage: Optional[str] = None):
        """Initialize and load dataset.

        Set variables: `self._data_train`, `self._data_val`, `self._data_test`.
        """

        # load datasets only if they're not loaded already
        if not self._data_train or not self._data_val or not self._data_test:
            if stage == 'train':
                self._data_train = hydra.utils.instantiate(
                    self._loader_cls['dataloader'], dataset_dir=self.hparams.data_dir,
                    stage='train', transform=self._train_transform)
                if len(self._data_train) == 0:
                    raise ValueError('Train dataset is empty.')
            elif stage in ['validation', 'test']:
                self._data_val = hydra.utils.instantiate(
                    self._loader_cls['dataloader'], dataset_dir=self.hparams.data_dir,
                    stage='validation', transform=self._test_transform)
                if len(self._data_val) == 0:
                    raise ValueError('Validation dataset is empty.')
                self._data_test = hydra.utils.instantiate(
                    self._loader_cls['dataloader'], dataset_dir=self.hparams.data_dir,
                    stage='test', transform=self._test_transform)
                if len(self._data_test) == 0:
                    raise ValueError('Test dataset is empty.')
            elif stage == 'predict':
                self._data_predict = hydra.utils.instantiate(
                    self._loader_cls['dataloader'], dataset_dir=self.hparams.data_dir,
                    stage='predict', transform=self._test_transform)
                if len(self._data_predict) == 0:
                    raise ValueError('Predict dataset is empty.')

    def train_dataloader(self):
        """Get train dataloader."""
        return DataLoader(
            dataset=self._data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        """Get validation dataloader."""
        return DataLoader(
            dataset=self._data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        """Get test dataloader."""
        return DataLoader(
            dataset=self._data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self):
        """Get prediction dataloader."""
        return DataLoader(
            dataset=self._data_predict,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
