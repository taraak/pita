from torch import nn
import torch

class EnergyModel(nn.Module):
    def __init__(
        self, 
        score_net: nn.Module,
    ):
        super(EnergyModel, self).__init__()
        self.score_net = score_net

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # parametrize energy as <score_net(t, x), x>
        return torch.sum(self.score_net(t, x) * x, dim=1)

    def forward_score(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.score_net(t, x)