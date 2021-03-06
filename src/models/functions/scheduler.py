import torch


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
