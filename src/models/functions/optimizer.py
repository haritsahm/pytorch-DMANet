# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import TYPE_CHECKING, Any

import torch
import torch.optim

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any

def to_real(x):
    if torch.is_complex(x):
        return x.real
    else:
        return x


class DAdaptSGD(torch.optim.Optimizer):
    r"""
    Implements SGD with D-Adaptation automatic step-sizes. Leave LR set to 1 unless you encounter instability.
    Arguments:
        params (iterable): 
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): 
            Learning rate adjustment parameter. Increases or decreases the D-adapted learning rate.
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        momentum (float): 
            Momentum value in  the range [0,1) (default: 0.9).
        weight_decay (float): 
            Weight decay, i.e. a L2 penalty (default: 0).
        log_every (int): 
            Log using print every k steps, default 0 (no logging).
        d0 (float):
            Initial D estimate for D-adaptation (default 1e-6). Rarely needs changing.
        growth_rate (float):
            prevent the D estimate from growing faster than this multiplicative rate. 
            Default is inf, for unrestricted. More conservative values like 1.02 may
            help if training is unstable.
    """
    def __init__(self, params, 
        lr=1e-2, 
        momentum=0, 
        weight_decay=0, 
        log_every=0,
        d0=1e-6, growth_rate=float('inf')):
        defaults = dict(lr=lr,
            momentum=momentum, 
            weight_decay=weight_decay, k=0,
            log_every=log_every,
            gsq_weighted=0.0, d=d0, growth_rate=growth_rate)
        self.loggables = {}

        try:
            self.rank = torch.distributed.get_rank()
        except:
            self.rank = 0

        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        group = self.param_groups[0]
        lr = group['lr']
        decay = group['weight_decay']
        momentum = group['momentum']
        log_every = group['log_every']
        ck = 1 - momentum
        k = group['k']

        gsq_weighted = group['gsq_weighted']
        growth_rate = group['growth_rate']
        d = group['d']
        

        g_sq = 0.0
        sk_sq = 0.0

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                
                # Apply weight decay
                if decay != 0:
                    if grad.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients")

                    grad.add_(p.data, alpha=decay)

                state = self.state[p]

                g_sq += (grad * grad).sum().item()


        group = self.param_groups[0]
        if k == 0: 
            group['g0_norm'] = g0_norm = math.sqrt(g_sq)

        g0_norm = group['g0_norm']
        dlr = d*lr/g0_norm

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if 'z' not in state:
                    z = state['z'] = torch.clone(p.data).detach()
                    s = state['s'] = torch.zeros_like(p.data).detach()
                    x0 = state['x0'] = torch.clone(p.data).detach()

                s = state['s']
                s.data.add_(grad, alpha=dlr)
                sk_sq += (s * s).sum().item()
            ######

        gsq_weighted = group['gsq_weighted'] = gsq_weighted + dlr*dlr*g_sq
        d_hat = d

        if lr > 0.0:
            d_hat = (sk_sq - gsq_weighted)/(math.sqrt(sk_sq))
            d = group['d'] = max(d, min(d_hat, d*growth_rate))

        if log_every > 0 and k % log_every == 0:
            print(f"(r={self.rank},k={k}) dlr: {dlr} d_hat: {d_hat}, d: {d}. sk_sq={sk_sq} gsq_weighted={gsq_weighted} g0_norm={g0_norm}", flush=True)

        for group in self.param_groups:
            group['gsq_weighted'] = gsq_weighted
            group['d'] = d
            group['g0_norm'] = g0_norm
            ######################################
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                s = state['s']
                x0 = state['x0']
                z = state['z']

                # z step
                z.data.copy_(x0 - s)

                # x step
                p.data.mul_(1-ck).add_(z, alpha=ck)

            group['k'] = k + 1

        return loss


class DAdaptAdam(torch.optim.Optimizer):
    r"""
    Implements Adam with D-Adaptation automatic step-sizes. Leave LR set to 1 unless you encounter instability.
    Arguments:
        params (iterable): 
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): 
            Learning rate adjustment parameter. Increases or decreases the D-adapted learning rate.
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        momentum (float): 
            Momentum value in  the range [0,1) (default: 0.9).
        eps (float): 
            Term added to the denominator outside of the root operation to improve numerical stability. (default: 0).
        weight_decay (float): 
            Weight decay, i.e. a L2 penalty (default: 0).
        log_every (int): 
            Log using print every k steps, default 0 (no logging).
        decouple (boolean): 
            Use AdamW style decoupled weight decay
        d0 (float):
            Initial D estimate for D-adaptation (default 1e-6). Rarely needs changing.
        growth_rate (float):
            prevent the D estimate from growing faster than this multiplicative rate. 
            Default is inf, for unrestricted. Values like 1.02 give a kind of learning
            rate warmup effect.
    """
    def __init__(self, params, lr=1.0, 
                 betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, log_every=0,
                 decouple=False,
                 d0=1e-6, growth_rate=float('inf')):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        if decouple:
            print(f"Using decoupled weight decay")

        
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        d = d0, 
                        k=0, 
                        gsq_weighted=0.0,
                        log_every=log_every,
                        growth_rate=growth_rate,
                        decouple=decouple)
        
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return False

    @property
    def supports_flat_params(self):
        return True

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()


        g_sq = 0.0
        sksq_weighted = 0.0
        sk_l1 = 0.0

        ngroups = len(self.param_groups)

        group = self.param_groups[0]
        gsq_weighted = group['gsq_weighted']
        d = group['d']
        lr = group['lr']
        dlr = d*lr
        
        growth_rate = group['growth_rate']
        decouple = group['decouple']
        log_every = group['log_every']

        beta1, beta2 = group['betas']

        for group in self.param_groups:
            decay = group['weight_decay']
            k = group['k']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                
                # Apply weight decay (coupled variant)
                if decay != 0 and not decouple:
                    grad.add_(p.data, alpha=decay)

                state = self.state[p]

                # State initialization
                if 'step' not in state:
                    state['step'] = 0
                    state['s'] = torch.zeros_like(p.data, memory_format=torch.preserve_format).detach()
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format).detach()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(to_real(p.data), memory_format=torch.preserve_format).detach()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                grad_grad = to_real(grad * grad.conj())

                # Adam EMA updates
                exp_avg.mul_(beta1).add_(grad, alpha=dlr*(1-beta1))
                exp_avg_sq.mul_(beta2).add_(grad_grad, alpha=1-beta2)
                
                denom = exp_avg_sq.sqrt().add_(eps)

                g_sq += grad_grad.div_(denom).sum().item()

                s = state['s']
                s.mul_(beta2).add_(grad, alpha=dlr*(1-beta2))
                sksq_weighted += to_real(s * s.conj()).div_(denom).sum().item()
                sk_l1 += s.abs().sum().item()

            ######

        gsq_weighted = beta2*gsq_weighted + g_sq*(dlr**2)*(1-beta2)
        d_hat = d

        if lr > 0.0:
            d_hat = (sksq_weighted/(1-beta2) - gsq_weighted)/sk_l1
            d = max(d, min(d_hat, d*growth_rate))

        if log_every > 0 and k % log_every == 0:
            print(f"ng: {ngroups} lr: {lr} dlr: {dlr} d_hat: {d_hat}, d: {d}. sksq_weighted={sksq_weighted:1.1e} sk_l1={sk_l1:1.1e} gsq_weighted={gsq_weighted:1.1e}")

        for group in self.param_groups:
            group['gsq_weighted'] = gsq_weighted
            group['d'] = d

            decay = group['weight_decay']
            k = group['k']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1

                denom = exp_avg_sq.sqrt().add_(eps)
                denom = denom.type(p.type())

                # Apply weight decay (decoupled variant)
                if decay != 0 and decouple:
                    p.data.add_(p.data, alpha=-decay * dlr)


                ### Take step
                p.data.addcdiv_(exp_avg, denom, value=-1)

            group['k'] = k + 1

        return loss
