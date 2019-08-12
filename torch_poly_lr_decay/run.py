import torch

from torch_poly_lr_decay import PolynomialLRDecay


if __name__ == '__main__':
    v = torch.zeros(10)
    optim = torch.optim.SGD([v], lr=0.01)
    scheduler = PolynomialLRDecay(optim, max_decay_steps=19, end_learning_rate=0.0001, power=2.0)

    for epoch in range(1, 20):
        scheduler.step(epoch)

        print(epoch, optim.param_groups[0]['lr'])
