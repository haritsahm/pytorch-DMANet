from typing import Any, List

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

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
        alpha_weight: float = 1.0,
        lr: float = 0.005,
        weight_decay: float = 0.0005,
    ):
        super().__init__()

        # TODO: Find a way to log lr to logger

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        # TODO: Implement: Dice
        if criterion_type == 'cross_entropy':
            self._criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f'Criterion {criterion_type} is not available')

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        # self.val_acc = Accuracy()
        # self.test_acc = Accuracy()

        # for logging best so far validation accuracy
        # self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        images, gt_masks = batch
        logits = self.forward(images)
        return images, logits, gt_masks

    def training_step(self, batch: Any, batch_idx: int):
        images, logits, gt_masks = self.step(batch)

        pd_masks, mid_aux, high_aux = logits

        mid_aux = F.interpolate(mid_aux, size=tuple(images.shape[2:]), mode='bilinear')
        high_aux = F.interpolate(high_aux, size=tuple(images.shape[2:]), mode='bilinear')

        pri_loss = self._criterion(pd_masks, gt_masks)
        aux_mid_loss = self._criterion(mid_aux, gt_masks)
        aux_high_loss = self._criterion(high_aux, gt_masks)

        print(pri_loss, aux_mid_loss, aux_high_loss,
              (aux_mid_loss + aux_high_loss), self.hparams.alpha_weight)

        joint_loss = pri_loss + (aux_mid_loss + aux_high_loss) * self.hparams.alpha_weight
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

        # log val metrics
        loss = self._criterion(logits, gt_masks)

        # TODO: Compute segmentation metrics
        acc = 0.01  # self.val_acc(preds, targets)

        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        # TODO: Log random samples

        return {'loss': loss}

    def validation_epoch_end(self, outputs: List[Any]):
        # acc = self.val_acc.compute()  # get val accuracy from current epoch
        # self.val_acc_best.update(acc)
        # self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
        pass

    def test_step(self, batch: Any, batch_idx: int):
        images, logits, gt_masks = self.step(batch)

        # log test metrics
        # TODO: Compute segmentation metrics
        acc = 0.01  # self.test_acc(preds, targets)

        self.log('test/acc', acc, on_step=False, on_epoch=True)

        # TODO: Log random samples

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        # self.test_acc.reset()
        # self.val_acc.reset()
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization. Normally
        you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.SGD(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

        scheduler = None

        return {'optimizer': optimizer}
