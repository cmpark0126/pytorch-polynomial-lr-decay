import torch

from torch_poly_scheduler import PolynomialLRScheduler


if __name__ == '__main__':
    v = torch.zeros(10)
    optim = torch.optim.SGD([v], lr=0.01)
    scheduler = PolynomialLRScheduler(optim, max_decay_steps=19, power=2.0)

    for epoch in range(1, 20):
        scheduler.step(epoch)

        print(epoch, optim.param_groups[0]['lr'])
