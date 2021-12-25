from torch.optim.lr_scheduler import _LRScheduler


class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        warmup_steps: linear preheat learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """

    def __init__(self, optimizer, max_decay_steps, warmup_steps=0, end_learning_rate=0.0001, power=1.0):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.warmup_steps = warmup_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)

    def get_lr(self):
        if self.warmup_steps > 0 and self.last_step <= self.warmup_steps:
            f = self.last_step / self.warmup_steps
            return [f * base_lr for base_lr in self.base_lrs]
        elif self.last_step <= self.max_decay_steps:
            return [(base_lr - self.end_learning_rate) *
                    ((1 - self.last_step / self.max_decay_steps) ** (self.power)) +
                    self.end_learning_rate for base_lr in self.base_lrs]
        else:
            return [self.end_learning_rate for _ in self.base_lrs]

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
