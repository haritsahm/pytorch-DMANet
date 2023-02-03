import torch

class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    """Decays the learning rate of each parameter group using a polynomial function
    in the given total_iters. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_iters (int): The number of steps that the scheduler decays the learning rate. Default: 5.
        power (int): The power of the polynomial. Default: 1.0.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.001 for all groups
        >>> # lr = 0.001     if epoch == 0
        >>> # lr = 0.00075   if epoch == 1
        >>> # lr = 0.00050   if epoch == 2
        >>> # lr = 0.00025   if epoch == 3
        >>> # lr = 0.0       if epoch >= 4
        >>> # xdoctest: +SKIP("undefined vars")
        >>> scheduler = PolynomialLR(self.opt, total_iters=4, power=1.0)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """
    def __init__(self, optimizer, total_iters=5, power=1.0, last_epoch=-1, verbose=False):
        self.total_iters = total_iters
        self.power = power
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        decay_factor = (1.0 - self.last_epoch / self.total_iters)
        decay_factor = (decay_factor / (1.0 - (self.last_epoch - 1) / self.total_iters)) ** self.power

        return [group["lr"] * decay_factor for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [
            (
                base_lr * (1.0 - min(self.total_iters, self.last_epoch) / self.total_iters) ** self.power
            )
            for base_lr in self.base_lrs
        ]


class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    """Plynomial Learning Rate Scheduler with Warmup.

    Apply the polynomial learning rate scheduler with warmup.
    Source:
    https://github.com/Tramac/awesome-semantic-segmentation-pytorch/blob/master/core/utils/lr_scheduler.py

    Parameters
    ----------
    optimizer : torch.optim
        torch optimizer module
    target_lr : int, optional
        Target learning rate, i.e. the ending learning rate, by default 0
    max_iters : int, optional
        Maximum number of iterations to be scheduled, by default 0
    power : float, optional
        Power factor, by default 0.9
    warmup_factor : float, optional
        Warmup factor, by default 1.0/3
    warmup_iters : int, optional
        Number of iterations for warmup, by default 500
    warmup_method : str, optional
        Warmup method, by default 'linear'
    last_epoch : int, optional
        _description_, by default -1

    Raises
    ------
    ValueError
        If warmup_method not 'constant' or 'linear'
    """

    def __init__(self,
                 optimizer: torch.optim,
                 target_lr: int = 0,
                 max_iters: int = 0,
                 power: float = 0.9,
                 warmup_factor: float = 1.0 / 3,
                 warmup_iters: int = 500,
                 warmup_method: str = 'linear',
                 last_epoch: int = -1):

        if warmup_method not in ('constant', 'linear'):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted "
                'got {}'.format(warmup_method))

        self.target_lr = target_lr
        self.max_iters = max_iters
        self.power = power
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method

        super(WarmupPolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        N = self.max_iters - self.warmup_iters
        T = self.last_epoch - self.warmup_iters
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor
            elif self.warmup_method == 'linear':
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            else:
                raise ValueError('Unknown warmup type.')
            return [self.target_lr + (base_lr - self.target_lr) * warmup_factor for base_lr in self.base_lrs]
        factor = pow(1 - T / N, self.power)
        return [self.target_lr + (base_lr - self.target_lr) * factor for base_lr in self.base_lrs]
