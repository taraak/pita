import math
import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch_cluster import radius_graph
from torch_geometric.nn import MessagePassing


class NeighborEmbedding(MessagePassing):
    def __init__(self, hidden_channels, num_rbf, cutoff_lower, cutoff_upper, max_z=100):
        super().__init__(aggr="add")
        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.distance_proj = nn.Linear(num_rbf, hidden_channels)
        self.combine = nn.Linear(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        nn.init.xavier_uniform_(self.distance_proj.weight)
        nn.init.xavier_uniform_(self.combine.weight)
        self.distance_proj.bias.data.fill_(0)
        self.combine.bias.data.fill_(0)

    def forward(self, z, x, edge_index, edge_weight, edge_attr):
        # remove self loops
        mask = edge_index[0] != edge_index[1]
        if not mask.all():
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_attr = edge_attr[mask]

        C = self.cutoff(edge_weight)
        W = self.distance_proj(edge_attr) * C.view(-1, 1)

        x_neighbors = self.embedding(z)
        # propagate_type: (x: Tensor, W: Tensor)
        x_neighbors = self.propagate(edge_index, x=x_neighbors, W=W, size=None)
        x_neighbors = self.combine(torch.cat([x, x_neighbors], dim=1))
        return x_neighbors

    def message(self, x_j, W):
        return x_j * W


class GaussianSmearing(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True):
        super().__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        offset, coeff = self._initial_params()
        if trainable:
            self.register_parameter("coeff", nn.Parameter(coeff))
            self.register_parameter("offset", nn.Parameter(offset))
        else:
            self.register_buffer("coeff", coeff)
            self.register_buffer("offset", offset)

    def _initial_params(self):
        offset = torch.linspace(self.cutoff_lower, self.cutoff_upper, self.num_rbf)
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def reset_parameters(self):
        offset, coeff = self._initial_params()
        self.offset.data.copy_(offset)
        self.coeff.data.copy_(coeff)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ExpNormalSmearing(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True):
        super().__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class CosineCutoff(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super().__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances):
        # DEBUG
        # nan_count = torch.isnan(distances).sum()
        # inf_count = torch.isinf(distances).sum()
        # Log counts without control flow
        # print("NaNs in distances:", nan_count)
        # print("Infs in distances:", inf_count)
        # print("distances min/max:", distances.min(), distances.max())

        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            return cutoffs


class Distance(nn.Module):
    def __init__(
        self,
        cutoff_lower,
        cutoff_upper,
        max_num_neighbors=32,
        return_vecs=False,
        loop=False,
    ):
        super().__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_num_neighbors = max_num_neighbors
        self.return_vecs = return_vecs
        self.loop = loop

    def forward(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ):
        if edge_index is None:
            edge_index = radius_graph(
                pos,
                r=self.cutoff_upper,
                batch=batch,
                loop=self.loop,
                max_num_neighbors=self.max_num_neighbors + 1,
            )

        # make sure we didn't miss any neighbors due to max_num_neighbors
        assert not (
            torch.unique(edge_index[0], return_counts=True)[1] > self.max_num_neighbors
        ).any(), (
            "The neighbor search missed some atoms due to "
            "max_num_neighbors being too low. Please increase "
            "this parameter to include the maximum number of atoms within the cutoff."
        )

        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        mask: Optional[torch.Tensor] = None
        if self.loop:
            # mask out self loops when computing distances because
            # the norm of 0 produces NaN gradients
            # NOTE: might influence force predictions as self loop gradients are ignored
            mask = edge_index[0] != edge_index[1]
            edge_weight = torch.zeros(edge_vec.size(0), device=edge_vec.device)
            edge_weight[mask] = torch.norm(edge_vec[mask], dim=-1)
        else:
            edge_weight = torch.norm(edge_vec, dim=-1)

        lower_mask = edge_weight >= self.cutoff_lower
        if self.loop and mask is not None:
            # keep self loops even though they might be below the lower cutoff
            lower_mask = lower_mask | ~mask
        edge_index = edge_index[:, lower_mask]
        edge_weight = edge_weight[lower_mask]

        if self.return_vecs:
            edge_vec = edge_vec[lower_mask]
            return edge_index, edge_weight, edge_vec
        # TODO: return only `edge_index` and `edge_weight` once
        # Union typing works with TorchScript (https://github.com/pytorch/pytorch/pull/53180)
        return edge_index, edge_weight, None


class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
        self,
        hidden_channels,
        out_channels,
        intermediate_channels=None,
        activation="silu",
        scalar_activation=False,
        vector_output=False,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.vector_output = vector_output
        self.layer_norm = layer_norm

        proj_out_channels = out_channels
        if self.vector_output:
            proj_out_channels = 1

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, proj_out_channels, bias=False)

        act_class = act_class_mapping[activation]

        if vector_output:
            self.update_net = nn.Sequential(
                nn.Linear(hidden_channels * 2, intermediate_channels),
                act_class(),
                nn.Linear(intermediate_channels, out_channels + 1),
            )
        else:
            self.update_net = nn.Sequential(
                nn.Linear(hidden_channels * 2, intermediate_channels),
                act_class(),
                nn.Linear(intermediate_channels, out_channels * 2),
            )
        if layer_norm:
            # add a layer norm after first linear layer
            self.update_net = nn.Sequential(
                self.update_net[0],
                nn.LayerNorm(intermediate_channels),
                self.update_net[1],
                self.update_net[2],
            )

        self.act = act_class() if scalar_activation else None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[-1].weight)
        self.update_net[-1].bias.data.fill_(0)

    def forward(self, x, v):
        vec1_buffer = self.vec1_proj(v)

        # detach zero-entries to avoid NaN gradients during force loss backpropagation
        vec1 = torch.zeros(vec1_buffer.size(0), vec1_buffer.size(2), device=vec1_buffer.device)

        # mask = (vec1_buffer != 0).view(vec1_buffer.size(0), -1).any(dim=1)
        mask = (vec1_buffer != 0).view(vec1_buffer.size(0), -1).all(dim=1)
        # if not mask.all():
        #     warnings.warn(
        #         (
        #             f"Skipping gradients for {(~mask).sum()} atoms due to "
        #             "vector features being zero. This is likely due to atoms "
        #             "being outside the cutoff radius of any other atom. "
        #             "These atoms will not interact with any other atom "
        #             "unless you change the cutoff."
        #         )
        #     )
        # num_skipped = (~mask).sum()
        # print("Num atoms skipped due to cutoff mask:", num_skipped)        # Only log tensor value safely

        # warnings.warn(
        #     f"Skipping gradients for {num_skipped} atoms due to "
        #     "vector features being zero. This is likely due to atoms "
        #     "being outside the cutoff radius of any other atom. "
        #     "These atoms will not interact with any other atom "
        #     "unless you change the cutoff."
        # ) if num_skipped > 0 else None

        # vec1[mask] = torch.norm(vec1_buffer[mask], dim=-2)
        vec1 = torch.where(mask.unsqueeze(-1), torch.norm(vec1_buffer, dim=-2), vec1)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)

        if self.vector_output:
            out = self.update_net(x)
            x, v = out[:, : self.out_channels], out[:, self.out_channels :]
        else:
            x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)

        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v


rbf_class_mapping = {"gauss": GaussianSmearing, "expnorm": ExpNormalSmearing}

act_class_mapping = {
    "ssp": ShiftedSoftplus,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}
