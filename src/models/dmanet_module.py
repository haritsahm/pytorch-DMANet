from typing import Any, List

import aim
import neptune.new as neptune
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics as tm
from neptune.new.types import File
from pytorch_lightning import LightningModule

import src.models.functions.scheduler as lr_scheduler
from src.utils import visualize


class DMANetLitModule(LightningModule):
    """LightningModule for DMA Network.

    For full documentation of LightningModule, plese read the docs.
    Source: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)
    """

    def __init__(
        self,
        net: torch.nn.Module,
        criterion_type: str = 'cross_entropy',
        aux_weight: float = 1.0,
        lr: float = 0.005,
        weight_decay: float = 0.0005,
        warmup_iters: int = 2500,
    ):
        super().__init__()

        # TODO: Find a way to log lr to logger

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self._net = net

        # TODO: Implement: Dice loss
        if criterion_type == 'cross_entropy':
            self._criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f'Criterion {criterion_type} is not available')

        self._val_metrics = tm.MetricCollection([
            tm.Accuracy(num_classes=self._net.num_classes,
                        average='macro', mdmc_average='samplewise', multiclass=True),
            tm.JaccardIndex(num_classes=self._net.num_classes),
            tm.F1Score(num_classes=self._net.num_classes, average='macro',
                       mdmc_average='samplewise', multiclass=True)
        ])
        self._test_metrics = tm.MetricCollection([
            tm.Accuracy(num_classes=self._net.num_classes,
                        average='macro', mdmc_average='samplewise', multiclass=True),
            tm.JaccardIndex(num_classes=self._net.num_classes),
            tm.F1Score(num_classes=self._net.num_classes, average='macro',
                       mdmc_average='samplewise', multiclass=True)
        ])

    def forward(self, x: torch.Tensor):
        return self._net(x)

    def step(self, batch: Any):
        images, gt_masks = batch
        logits = self.forward(images)
        return images, logits, gt_masks

    def training_step(self, batch: Any, batch_idx: int):
        images, logits, gt_masks = self.step(batch)

        pd_masks, mid_aux, high_aux = logits

        mid_aux = F.interpolate(mid_aux, size=tuple(
            images.shape[2:]), mode='bilinear', align_corners=True)
        high_aux = F.interpolate(high_aux, size=tuple(
            images.shape[2:]), mode='bilinear', align_corners=True)

        pri_loss = self._criterion(pd_masks, gt_masks)
        aux_mid_loss = self._criterion(mid_aux, gt_masks)
        aux_high_loss = self._criterion(high_aux, gt_masks)

        joint_loss = pri_loss + (aux_mid_loss + aux_high_loss) * self.hparams.aux_weight
        joint_loss.retain_grad()

        self.log('train/loss', joint_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/pri_loss', pri_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/aux_mid_loss', aux_mid_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/aux_high_loss', aux_high_loss, on_step=False, on_epoch=True, prog_bar=False)

        # we can return here dict with any tensors
        return {'loss': joint_loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        images, logits, gt_masks = self.step(batch)

        # log val loss
        loss = self._criterion(logits, gt_masks)

        # log val metrics
        pd_masks = torch.argmax(logits, dim=1)

        self._val_metrics(pd_masks, gt_masks)

        if batch_idx % 100 == 0 and torch.rand(1).item() > 0.7:
            images = images.cpu().numpy()
            pd_masks = pd_masks.to(torch.uint8).cpu().numpy()

            for idx, (image, target) in enumerate(zip(images, pd_masks)):

                image = (image * 255).astype(np.uint8).transpose((1, 2, 0))
                colored_mask = visualize.show_prediction(image, target, overlay=0.5)

                for logger in self.loggers:
                    if isinstance(logger.experiment, neptune.Run):
                        logger.experiment['val/predictions'].log(File.as_image(colored_mask))
                    if isinstance(logger.experiment, aim.Run):
                        log_image = aim.Image(colored_mask, format='jpeg', optimize=True, quality=75)
                        logger.experiment.track(log_image, name='images',
                                                epoch=self.current_epoch, context={'subset': 'val'})

        return {'loss': loss}

    def validation_epoch_end(self, outputs: List[Any]):
        val_metrics = self._val_metrics.compute()
        # self.log('val/avg_loss', outputs["loss"], on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/acc', val_metrics['Accuracy'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/mIoU', val_metrics['JaccardIndex'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/F1Score', val_metrics['F1Score'], on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        images, logits, gt_masks = self.step(batch)

        pd_masks = torch.argmax(logits, dim=1)

        if batch_idx % 100 == 0 and torch.rand(1).item() > 0.7:
            images = images.cpu().numpy()
            pd_masks = pd_masks.to(torch.uint8).cpu().numpy()

            for idx, (image, target) in enumerate(zip(images, pd_masks)):

                image = (image * 255).astype(np.uint8).transpose((1, 2, 0))
                colored_mask = visualize.show_prediction(image, target, overlay=0.5)

                for logger in self.loggers:
                    if isinstance(logger.experiment, neptune.Run):
                        logger.experiment['test/predictions'].log(File.as_image(colored_mask))
                    if isinstance(logger.experiment, aim.Run):
                        log_image = aim.Image(colored_mask, format='jpeg', optimize=True, quality=75)
                        logger.experiment.track(log_image, name='images',
                                                epoch=self.current_epoch, context={'subset': 'test'})

        # TODO: Save output to some standards JSON/txt

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        """Reset metrics at the end of every epoch."""

        self._val_metrics.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization. Normally
        you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.SGD(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

        scheduler = lr_scheduler.WarmupPolyLR(
            optimizer=optimizer, target_lr=self.hparams.lr, max_iters=self.trainer.estimated_stepping_batches,
            warmup_iters=self.hparams.warmup_iters)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
