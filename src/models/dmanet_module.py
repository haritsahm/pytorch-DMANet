from typing import Any, List

import torch
import torch.nn.functional as F
import torchmetrics as tm
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

import src.models.functions.scheduler as lr_scheduler
from src.models.components.dma_net import DMANet


class DMANetLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    # TODO: Docstring

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
    ):
        super().__init__()

        # TODO: Find a way to log lr to logger

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self._net = net

        # loss function
        # TODO: Implement: Dice
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
        self.log('train/aux_high_loss', aux_high_loss,
                 on_step=False, on_epoch=True, prog_bar=False)

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

        # TODO: Compute segmentation metrics
        self._val_metrics(pd_masks, gt_masks)

        # TODO: Log random samples

        return {'loss': loss}

    def validation_epoch_end(self, outputs: List[Any]):
        val_metrics = self._val_metrics.compute()
        # self.log('val/avg_loss', outputs["loss"], on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/acc', val_metrics['Accuracy'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/mIoU', val_metrics['JaccardIndex'],
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/F1Score', val_metrics['F1Score'],
                 on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        images, logits, gt_masks = self.step(batch)

        # log test metrics
        pd_masks = torch.argmax(logits, dim=1)

        # TODO: Compute segmentation metrics
        self._test_metrics(pd_masks, gt_masks)

        # TODO: Log random samples

    def test_epoch_end(self, outputs: List[Any]):
        test_metrics = self._test_metrics.compute()
        self.log('test/acc', test_metrics['Accuracy'], on_step=False, on_epoch=True)
        self.log('test/mIoU', test_metrics['JaccardIndex'],
                 on_step=False, on_epoch=True)
        self.log('test/F1Score', test_metrics['F1Score'],
                 on_step=False, on_epoch=True)

    def on_epoch_end(self):
        """Reset metrics at the end of every epoch."""

        self._val_metrics.reset()
        self._test_metrics.reset()

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
            optimizer=optimizer, target_lr=self.hparams.lr, max_iters=self.trainer.estimated_stepping_batches, warmup_iters=1500)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
