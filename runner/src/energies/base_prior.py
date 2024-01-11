import torch


class Prior:
    def __init__(self, dim, var, device="cpu"):
        self.dim = dim
        self.var = var
        self.dist = torch.distributions.MultivariateNormal(
            torch.zeros(dim).to(device), torch.eye(2).to(device) * var
        )

    def log_prior(self, x):
        return self.dist.log_prob(x)
