# pytorch-polynomial-lr-decay
Polynomial Learning Rate Decay Scheduler for PyTorch

This scheduler is frequently used in many DL paper. But there is no official implementation in PyTorch. So I propose this code.

## Install

```
$ pip install git+https://github.com/cmpark0126/pytorch-polynomial-lr-decay.git
```

## Usage

```python
from torch_poly_scheduler import PolynomialLRScheduler

scheduler_poly_lr_decay = PolynomialLRScheduler(optim, max_decay_steps=100, power=2.0)

for epoch in range(train_epoch):
    scheduler_poly_lr_decay.step()     # you can handle step as epoch number
    ...
```

or

```python
from torch_poly_scheduler import PolynomialLRScheduler

scheduler_poly_lr_decay = PolynomialLRScheduler(optim, max_decay_steps=100, power=2.0)

...

for batch_idx, (inputs, targets) in enumerate(trainloader):
    scheduler_poly_lr_decay.step()     # also, you can handle step as each iter number
```
