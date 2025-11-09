import torch
import torch.nn as nn


class SDFMLP(nn.Module):
    def __init__(self, hidden=64, layers=4):
        super().__init__()
        dims = [3] + [hidden] * (layers - 1) + [1]
        net = []
        for i in range(len(dims) - 2):
            net.append(nn.Linear(dims[i], dims[i + 1]))
            net.append(nn.Softplus(beta=100))
        net.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)
        d_output = torch.ones_like(y, device=y.device)
        grad = torch.autograd.grad(y, x, d_output, create_graph=True, retain_graph=True, only_inputs=True)[0]
        return y, grad
