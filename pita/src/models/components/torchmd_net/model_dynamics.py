from typing import Optional, Tuple

import numpy as np
import torch
from src.utils.data_utils import remove_mean
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

from .modules import CoorsNorm, EquivariantVectorOutput
from .utils import CosineCutoff, NeighborEmbedding, act_class_mapping, rbf_class_mapping


def center(pos, batch):
    pos_center = pos - scatter(pos, batch, dim=0, reduce="mean")[batch]
    return pos_center


class EquivariantMultiHeadAttention(MessagePassing):
    def __init__(
        self,
        n_particles: int,
        hidden_channels: int,
        num_rbf: int,
        distance_influence: str,
        num_heads: int,
        activation: str,
        attn_activation: str,
        cutoff_lower: float,
        cutoff_upper: float,
        node_attr_dim: int = 0,
        qk_norm: bool = False,
        norm_coors: bool = False,
        norm_coors_scale_init: float = 1e-2,
        so3_equivariant: bool = False,
    ):
        super().__init__(aggr="add", node_dim=0)
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.n_particles = n_particles
        self.so3_equivariant = so3_equivariant
        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads

        self.layernorm = nn.LayerNorm(hidden_channels)
        self.node_attr_dim = node_attr_dim
        self.norm_coors = norm_coors  # boolean
        self.coors_norm = (
            CoorsNorm(scale_init=norm_coors_scale_init) if norm_coors else nn.Identity()
        )
        self.act = activation()
        self.attn_activation = act_class_mapping[attn_activation]()
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)
        self.qk_norm = qk_norm

        # input_channels = (
        #     hidden_channels + 1 + (hidden_channels if node_attr_dim > 0 else 0)
        # )
        input_channels = hidden_channels + (hidden_channels if node_attr_dim > 0 else 0)
        # print(f"input_channels:{input_channels}")
        self.mixing_mlp = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        if qk_norm:
            # add layer norm to q and k projections
            # based on https://arxiv.org/pdf/2302.05442.pdf
            self.q_proj = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
            )
            self.k_proj = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
            )
        else:
            self.q_proj = nn.Linear(hidden_channels, hidden_channels)
            self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels * (3 + int(so3_equivariant)))
        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 3, bias=False)

        # projection linear layers for edge attributes
        self.dk_proj = nn.Linear(num_rbf, hidden_channels)
        self.dv_proj = nn.Linear(num_rbf, hidden_channels * (3 + int(so3_equivariant)))

        self.reset_parameters()

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        if self.qk_norm:
            self.q_proj[0].bias.data.fill_(0)
            nn.init.xavier_uniform_(self.q_proj[0].weight)
            self.k_proj[0].bias.data.fill_(0)
            nn.init.xavier_uniform_(self.k_proj[0].weight)
        else:
            self.q_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.q_proj.weight)
            self.k_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.k_proj.weight)

        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.vec_proj.weight)
        if self.dk_proj:
            nn.init.xavier_uniform_(self.dk_proj.weight)
            self.dk_proj.bias.data.fill_(0)
        if self.dv_proj:
            nn.init.xavier_uniform_(self.dv_proj.weight)
            self.dv_proj.bias.data.fill_(0)

    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij, node_attr):
        # Mix x with node_attr(time, beta)
        x = self.mixing_mlp(torch.cat([x, node_attr], dim=1))
        # zeros | (h_z), h_t,h_beta OR embedding(z), h_t,h_beta

        # Input features: (BSxnum_atoms, hidden_channels)
        x = self.layernorm(x)
        # key/query features: (BSxnum_atoms, num_heads, head_dim)
        # where head_dim * num_heads == hidden_channels
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        # value features: (BSxnum_atoms, num_heads, 3 * head_dim)
        v = self.v_proj(x).reshape(
            -1, self.num_heads, self.head_dim * (3 + int(self.so3_equivariant))
        )

        # vec features: (BSxnum_atoms, 3, hidden_channels) (all invariant)
        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec = vec.reshape(-1, 3, self.num_heads, self.head_dim)
        vec_dot = (vec1 * vec2).sum(dim=1)

        # transform edge attributes (relative distances and user provided edge attributes)
        # into dk and dv vectors with shape (num_edges, num_heads, head_dim)
        # and (num_edges, num_heads, 3 * head_dim) respectively
        dk = self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
        dv = self.act(self.dv_proj(f_ij)).reshape(
            -1, self.num_heads, self.head_dim * (3 + int(self.so3_equivariant))
        )

        if isinstance(edge_index, list):  # If it's a list
            edge_index = torch.stack(edge_index, dim=0)  # Convert to [2, num_edges] tensor
        # Message Passing Propagate
        x, vec = self.propagate(
            edge_index,  # (2, edges)
            q=q,
            k=k,
            v=v,
            vec=vec,
            dk=dk,
            dv=dv,
            r_ij=r_ij,
            d_ij=d_ij,
            size=None,
        )
        # new shape: (BSxnum_atoms, hidden_channels)
        x = x.reshape(-1, self.hidden_channels)
        # new shape: (BSxnum_atoms, 3, hidden_channels)
        vec = vec.reshape(-1, 3, self.hidden_channels)
        # normalize the vec if norm_coors is True
        vec = self.coors_norm(vec)

        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dvec = vec3 * o1.unsqueeze(1) + vec
        dx = vec_dot * o2 + o3
        return dx, dvec

    def message(
        self,
        q_i: Tensor,  # (num_edges, num_heads, head_dim)
        k_j: Tensor,  # (num_edges, num_heads, head_dim)
        v_j: Tensor,  # (num_edges, num_heads, head_dim * 3)
        vec_j: Tensor,  # (num_edges, 3, num_heads, head_dim)
        dk: Tensor,  # (num_edges, num_heads, head_dim)
        dv: Tensor,  # (num_edges, num_heads, head_dim * 3)
        r_ij: Tensor,  # (num_edges,) edge distances
        d_ij: Tensor,  # (num_edges, 3) edge vectors (unit vectors)
    ):
        # dot product attention, a score for each edge
        attn = (q_i * k_j * dk).sum(dim=-1)  # (num_edges, num_heads)

        # apply attention activation function
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)

        # value pathway
        v_j = v_j * dv  # multiply with edge attr features

        if self.so3_equivariant:
            x, vec1, vec2, vec3 = torch.split(v_j, self.head_dim, dim=2)
        else:
            x, vec1, vec2 = torch.split(v_j, self.head_dim, dim=2)
            vec3 = None

        # update scalar features
        x = x * attn.unsqueeze(2)  # (num_edges, num_heads, head_dim)
        # update vector features (num_edges, 3, num_heads, head_dim)
        if self.so3_equivariant:
            vec = (
                vec_j * vec1.unsqueeze(1)
                + vec2.unsqueeze(1) * d_ij.unsqueeze(2).unsqueeze(3)
                + vec3.unsqueeze(1) * torch.cross(d_ij.unsqueeze(2).unsqueeze(3), vec_j, dim=1)
            )
        else:
            vec = vec_j * vec1.unsqueeze(1) + vec2.unsqueeze(1) * d_ij.unsqueeze(2).unsqueeze(3)
        return x, vec

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        # scatter edge-level features (for x and vec) to node-level
        # x shape: (num_atoms, num_heads, head_dim)
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        # vec shape: (num_atoms, 3, num_heads, head_dim)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs


class TorchMD_ET_dynamics(nn.Module):
    r"""The TorchMD equivariant Transformer architecture.

    Parameters
    ----------
    hidden_channels (int, optional): Hidden embedding size.
        (default: :obj:`128`)
    num_layers (int, optional): The number of attention layers.
        (default: :obj:`6`)
    num_rbf (int, optional): The number of radial basis functions :math:`\mu`.
        (default: :obj:`50`)
    rbf_type (string, optional): The type of radial basis function to use.
        (default: :obj:`"expnorm"`)
    trainable_rbf (bool, optional): Whether to train RBF parameters with
        backpropagation. (default: :obj:`True`)
    activation (string, optional): The type of activation function to use.
        (default: :obj:`"silu"`)
    attn_activation (string, optional): The type of activation function to use
        inside the attention mechanism. (default: :obj:`"silu"`)
    neighbor_embedding (bool, optional): Whether to perform an initial neighbor
        embedding step. (default: :obj:`True`)
    num_heads (int, optional): Number of attention heads.
        (default: :obj:`8`)
    distance_influence (string, optional): Where distance information is used inside
        the attention mechanism. (default: :obj:`"both"`)
    cutoff_lower (float, optional): Lower cutoff distance for interatomic interactions.
        (default: :obj:`0.0`)
    cutoff_upper (float, optional): Upper cutoff distance for interatomic interactions.
        (default: :obj:`5.0`)
    max_z (int, optional): Maximum atomic number. Used for initializing embeddings.
        (default: :obj:`100`)
    qk_norm (bool, optional):
        Applies layer norm to q and k projections. Supposed to
        stabilize the training based on
        https://arxiv.org/pdf/2302.05442.pdf. (default: :obj:`False`)
    """

    def __init__(
        self,
        n_particles: int,
        hidden_channels: int = 128,
        num_layers: int = 6,
        num_rbf: int = 50,
        rbf_type: str = "expnorm",
        trainable_rbf: bool = True,
        activation: str = "silu",
        attn_activation: str = "silu",
        neighbor_embedding: bool = False,
        num_heads: int = 8,
        distance_influence: str = "both",
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 10.0,
        max_z: int = 100,
        node_attr_dim: int = 2,  # 2
        edge_attr_dim: int = 1,  # 1
        qk_norm: bool = False,
        norm_coors: bool = False,
        norm_coors_scale_init: float = 1e-2,
        clip_during_norm: bool = False,
        so3_equivariant: bool = False,
        condition_time: bool = True,
        condition_temperature: bool = True,
    ):
        super().__init__()

        assert distance_influence in ["keys", "values", "both", "none"]
        assert rbf_type in rbf_class_mapping, (
            f'Unknown RBF type "{rbf_type}". '
            f'Choose from {", ".join(rbf_class_mapping.keys())}.'
        )
        assert activation in act_class_mapping, (
            f'Unknown activation function "{activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )
        assert attn_activation in act_class_mapping, (
            f'Unknown attention activation function "{attn_activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )

        self.n_particles = n_particles
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.attn_activation = attn_activation
        self.neighbor_embedding = neighbor_embedding
        self.num_heads = num_heads
        self.distance_influence = distance_influence
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_z = max_z
        self.node_attr_dim = node_attr_dim
        self.edge_attr_dim = edge_attr_dim
        self.clip_during_norm = clip_during_norm
        self.condition_time = condition_time
        self.conition_temperature = condition_temperature

        act_class = act_class_mapping[activation]

        # self.embedding = nn.Embedding(self.max_z, self.hidden_channels)
        self.embedding = None
        # self.neighbor_embedding = (
        #     NeighborEmbedding(
        #         hidden_channels,
        #         num_rbf + edge_attr_dim,
        #         cutoff_lower,
        #         cutoff_upper,
        #         self.max_z,
        #     )
        #     if neighbor_embedding
        #     else None
        # )
        self.neighbor_embedding = None

        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )
        if self.node_attr_dim > 0:
            self.node_mlp = nn.Sequential(
                nn.Linear(node_attr_dim, hidden_channels),
                act_class(),
                nn.LayerNorm(hidden_channels),
                nn.Linear(hidden_channels, hidden_channels),
            )

        self.attention_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = EquivariantMultiHeadAttention(
                n_particles,
                hidden_channels,
                num_rbf + edge_attr_dim,
                distance_influence,
                num_heads,
                act_class,
                attn_activation,
                cutoff_lower,
                cutoff_upper,
                node_attr_dim=node_attr_dim,
                qk_norm=qk_norm,
                norm_coors=norm_coors,
                norm_coors_scale_init=norm_coors_scale_init,
                so3_equivariant=so3_equivariant,
            )  # .jittable() TODO: Removing for now
            self.attention_layers.append(layer)

        self.out_norm = nn.LayerNorm(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        # self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        # if self.neighbor_embedding is not None:
        #     self.neighbor_embedding.reset_parameters()
        for attn in self.attention_layers:
            attn.reset_parameters()
        self.out_norm.reset_parameters()

    def forward(
        self,
        # t: Tensor,
        h: Tensor,
        pos: Tensor,
        # batch: Tensor,
        edge_index: Optional[Tensor] = None,
        # node_attr: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        edge_vec: Optional[Tensor] = None,
        z: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if self.node_attr_dim > 0:
            node_attr = self.node_mlp(h)
        else:
            node_attr = None

        # if torch.any(edge_index < 0) or torch.any(edge_index >= num_nodes):
        #     print("Invalid edge index detected!")
        #     print("edge_index min:", edge_index.min().item(), "max:", edge_index.max().item())
        #     print("num_nodes:", num_nodes)
        #     assert False  # force crash

        # edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        # edge_weight = (edge_vec**2).sum(dim=1, keepdim=False)

        # edge_weight = edge_attr.clone()
        # mask = edge_index[0] == edge_index[1]
        # masked_edge_weight = edge_weight.masked_fill(mask, 1).unsqueeze(1)
        edge_weight = edge_attr.clone().squeeze(-1)  # shape: [79872]
        mask = edge_index[0] == edge_index[1]  # shape: [79872]
        masked_edge_weight = edge_weight.masked_fill(mask, 1.0).unsqueeze(-1)  # shape: [79872, 1]

        # print(f"edge_weight.shape:{edge_weight.shape}, masked_edge_weight.shape:{masked_edge_weight.shape}, edge_vec.shape:{edge_vec.shape}")

        if (
            self.clip_during_norm
        ):  # clip edge_weight to avoid exploding values if two nodes are close
            masked_edge_weight = masked_edge_weight.clamp(min=1.0e-2)

        edge_vec = edge_vec / masked_edge_weight

        # update edge_attributes with user input if they are given
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(1)  # (num_edges, 1)
            edge_attr_cat = self.distance_expansion(edge_attr).squeeze(1)
            # print(f"edge_attr_cat.shape:{edge_attr_cat.shape}")
            edge_attr = torch.cat(
                [edge_attr_cat, edge_attr], dim=-1
            )  # (num_edges, num_rbf + edge_attr_dim)
        else:
            edge_attr = self.distance_expansion(edge_attr)  # (num_edges, num_rbf)

        # embed atomic numbers using an embedding layer
        if z is not None:  # h_initial, already one-hot
            # if z.dim() > 1:
            #     z = z.squeeze()     # (num_atoms,)
            x = self.embedding(z)  # (BSxnum_atoms, self.hidden_channels)
        else:
            x = torch.zeros(pos.size(0), self.hidden_channels).to(pos.device)  # (BS, hidden_dim)
        # if self.neighbor_embedding is not None:
        #     x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)
        # vec here is invariant values, we are not modifying the vectors.
        # (BS x num_atoms, 3, hidden_channels)
        vec = torch.zeros(
            pos.size(0), 3, self.hidden_channels, device=pos.device
        )  # (BS*n_particles, n_dimension, hidden_dim) i.e. vector feat
        for attn in self.attention_layers:
            dx, dvec = attn(
                x,
                vec,
                edge_index,
                edge_weight,
                edge_attr,
                edge_vec,
                node_attr=node_attr,
                # t=t,
            )
            x = x + dx
            vec = vec + dvec
        x = self.out_norm(x)  # apply layer norm in the end.

        return x, vec, pos

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"num_rbf={self.num_rbf}, "
            f"rbf_type={self.rbf_type}, "
            f"trainable_rbf={self.trainable_rbf}, "
            f"activation={self.activation}, "
            f"attn_activation={self.attn_activation}, "
            f"neighbor_embedding={self.neighbor_embedding}, "
            f"num_heads={self.num_heads}, "
            f"distance_influence={self.distance_influence}, "
            f"cutoff_lower={self.cutoff_lower}, "
            f"cutoff_upper={self.cutoff_upper})"
        )


class TorchMDDynamics(nn.Module):
    r"""
    TorchMDDynamics Model for DDPM training.

    Parameters
    ----------
    hidden_channels (int, optional):
        Hidden embedding size. (default: :obj:`128`)
    num_layers (int, optional):
        The number of attention layers. (default: :obj:`8`)
    num_rbf (int, optional):
        The number of radial basis functions :math:`\mu`.
        (default: :obj:`64`)
    rbf_type (string, optional):
        The type of radial basis function to use.
        (default: :obj:`"expnorm"`)
    trainable_rbf (bool, optional):
        Whether to train RBF parameters with backpropagation.
        (default: :obj:`False`)
    activation (string, optional):
        The type of activation function to use. (default: :obj:`"silu"`)
    neighbor_embedding (bool, optional):
        Whether to perform an initial neighbor embedding step.
        (default: :obj:`True`)
    cutoff_lower (float, optional):
        Lower cutoff distance for interatomic interactions.
        (default: :obj:`0.0`)
    cutoff_upper (float, optional):
        Upper cutoff distance for interatomic interactions.
        (default: :obj:`5.0`)
    max_z (int, optional):
        Maximum atomic number. Used for initializing embeddings.
        (default: :obj:`100`)
    node_attr_dim (int, optional):
        Dimension of additional input node  features (non-atomic numbers).
    attn_activation (string, optional):
        The type of activation function to use inside the attention
        mechanism. (default: :obj:`"silu"`)
    num_heads (int, optional):
        Number of attention heads. (default: :obj:`8`)
    distance_influence (string, optional):
        Where distance information is used inside the attention
        mechanism. (default: :obj:`"both"`)
    qk_norm (bool, optional):
        Applies layer norm to q and k projections. Supposed to
        stabilize the training based on
        https://arxiv.org/pdf/2302.05442.pdf. (default: :obj:`False`)
    """

    def __init__(
        self,
        n_particles,
        n_dimension,
        hidden_nf: int = 32,
        n_layers: int = 3,
        num_rbf: int = 16,
        rbf_type: str = "expnorm",
        trainable_rbf: bool = False,
        activation: str = "silu",
        neighbor_embedding: int = True,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 10.0,
        max_z: int = 100,
        attn_activation: str = "silu",
        n_heads: int = 1,
        distance_influence: str = "both",
        reduce_op: str = "sum",
        qk_norm: bool = False,
        output_layer_norm: bool = True,
        clip_during_norm: bool = False,
        so3_equivariant: bool = False,
        condition_time=True,
        condition_temperature=True,
        is_alanine=True,
    ):
        super().__init__()
        self._n_particles = n_particles
        self._n_dimension = n_dimension
        self.edges = self._create_edges()
        self._edges_dict = {}
        self.condition_time = condition_time
        self.condition_temperature = condition_temperature
        self.is_alanine = is_alanine
        self.edge_attr_dim = 1

        if self.is_alanine:
            self.h_initial = self.get_h_initial()
            self.node_attr_dim = self.h_initial.shape[-1]
        else:
            self.node_attr_dim = 0

        if condition_time:
            self.node_attr_dim += 1
        if condition_temperature:
            self.node_attr_dim += 1

        self.representation_model = TorchMD_ET_dynamics(
            n_particles=self._n_particles,
            hidden_channels=hidden_nf,
            num_layers=n_layers,
            num_rbf=num_rbf,
            rbf_type=rbf_type,
            trainable_rbf=trainable_rbf,
            activation=activation,
            neighbor_embedding=neighbor_embedding,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            max_z=max_z,
            attn_activation=attn_activation,
            num_heads=n_heads,
            distance_influence=distance_influence,
            node_attr_dim=self.node_attr_dim,
            edge_attr_dim=self.edge_attr_dim,
            qk_norm=qk_norm,
            clip_during_norm=clip_during_norm,
            so3_equivariant=so3_equivariant,
            condition_time=condition_time,
            condition_temperature=condition_temperature,
        )
        self.output_model = EquivariantVectorOutput(
            hidden_channels=hidden_nf,
            activation=activation,
            reduce_op=reduce_op,
            layer_norm=output_layer_norm,
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()

    def _create_edges(self):
        rows, cols = [], []
        for i in range(self._n_particles):
            for j in range(i + 1, self._n_particles):
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
        return [torch.LongTensor(rows), torch.LongTensor(cols)]

    def _cast_edges2batch(self, edges, n_batch, n_nodes, device):
        if n_batch not in self._edges_dict:
            self._edges_dict = {}
            rows, cols = edges
            rows_total, cols_total = [], []
            for i in range(n_batch):
                rows_total.append(rows + i * n_nodes)
                cols_total.append(cols + i * n_nodes)
            rows_total = torch.cat(rows_total).to(device)
            cols_total = torch.cat(cols_total).to(device)

            self._edges_dict[n_batch] = [rows_total, cols_total]
        return self._edges_dict[n_batch]

    def get_h_initial(self):
        if self._n_particles == 22:
            atom_types = np.arange(22)
            atom_types[[1, 2, 3]] = 2
            atom_types[[19, 20, 21]] = 20
            atom_types[[11, 12, 13]] = 12
            return torch.nn.functional.one_hot(torch.tensor(atom_types))
        if self._n_particles == 33:
            atom_types = np.arange(33)
            atom_types[[1, 2, 3]] = 2
            atom_types[[9, 10, 11]] = 10
            atom_types[[19, 20, 21]] = 18
            atom_types[[29, 30, 31]] = 31
            h_initial = torch.nn.functional.one_hot(torch.tensor(atom_types))
            return h_initial
        if self._n_particles == 42:
            atom_types = np.arange(42)
            atom_types[[1, 2, 3]] = 2
            atom_types[[11, 12, 13]] = 12
            atom_types[[21, 22, 23]] = 22
            atom_types[[31, 32, 33]] = 32
            atom_types[[39, 40, 41]] = 40
            h_initial = torch.nn.functional.one_hot(torch.tensor(atom_types))
            return h_initial
        if self._n_particles == 55 or self._n_particles == 13:
            return torch.zeros(self._n_particles, 1)
        if self._n_particles >= 53:
            return self.get_hidden()

    def get_hidden(self):
        # n_encodings = 78
        amino_dict = {
            "ALA": 0,
            "ARG": 1,
            "ASN": 2,
            "ASP": 3,
            "CYS": 4,
            "GLN": 5,
            "GLU": 6,
            "GLY": 7,
            "HIS": 8,
            "ILE": 9,
            "LEU": 10,
            "LYS": 11,
            "MET": 12,
            "PHE": 13,
            "PRO": 14,
            "SER": 15,
            "THR": 16,
            "TRP": 17,
            "TYR": 18,
            "VAL": 19,
        }
        atom_types = []
        amino_idx = []
        amino_types = []
        for i, amino in enumerate(self.topology.residues):
            for atom_name in amino.atoms:
                amino_idx.append(i)
                amino_types.append(amino_dict[amino.name])
                if atom_name.name[0] == "H" and atom_name.name[-1] in ("1", "2", "3"):
                    if amino_dict[amino.name] in (8, 13, 17, 18) and atom_name.name[:2] in (
                        "HE",
                        "HD",
                        "HZ",
                        "HH",
                    ):
                        pass
                    else:
                        atom_name.name = atom_name.name[:-1]
                if atom_name.name[:2] == "OE" or atom_name.name[:2] == "OD":
                    atom_name.name = atom_name.name[:-1]
                atom_types.append(atom_name.name)
        atom_types_dict = np.array(
            [self.atom_types_encoding[atom_type] for atom_type in atom_types]
        )
        atom_onehot = torch.nn.functional.one_hot(
            torch.tensor(atom_types_dict), num_classes=len(self.atom_types_encoding)
        )
        if self._n_particles == 53:
            num_classes = 5
        elif self._n_particles == 63:
            num_classes = 6
        amino_idx_onehot = torch.nn.functional.one_hot(
            torch.tensor(amino_idx), num_classes=num_classes
        )
        amino_types_onehot = torch.nn.functional.one_hot(torch.tensor(amino_types), num_classes=20)

        h_initial = torch.cat([amino_idx_onehot, amino_types_onehot, atom_onehot], dim=1)
        return h_initial

    def forward(
        self,
        t: Tensor,
        pos: Tensor,
        beta: Tensor,
        # z: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass over torchmd-net model.

        Parameters
        ----------
        t: torch.Tensor
            Time steps of diffusion, shape (BS,)
        pos: torch.Tensor
            Atomic positions, shape (BS, num_atoms x 3)
        # z: torch.Tensor, optional
        #     Atomic numbers, shape (num_atoms,)
        """

        t = t.view(-1, 1)
        if t.numel() == 1:
            t = t.repeat(pos.shape[0], 1)

        n_batch = pos.shape[0]
        edges = self._cast_edges2batch(self.edges, n_batch, self._n_particles, device=pos.device)
        edges = [edges[0], edges[1]]

        # Changed by Leon
        pos_rs = pos.reshape(n_batch * self._n_particles, self._n_dimension).clone()

        if t.shape != (n_batch, 1):
            t = t.repeat(n_batch)
        t = t.repeat(1, self._n_particles)
        t = t.reshape(n_batch * self._n_particles, 1)

        if self.condition_temperature:
            beta = beta.view(-1, 1)
            beta = beta.repeat(1, self._n_particles)
            beta = beta.reshape(n_batch * self._n_particles, 1)
            t = torch.cat([t, beta], dim=-1)

        if self.is_alanine:
            h = self.h_initial.to(pos.device).reshape(1, -1)
            h = h.repeat(n_batch, 1)
            h = h.reshape(n_batch * self._n_particles, -1)
            h = torch.cat([h, t], dim=-1)
            """
            h = self.h_initial.to(pos.device).unsqueeze(0).expand(n_batch, -1, -1)
            h = torch.cat([h, h_t, h_beta], dim=-1)
            # alternatively,
            # z = self.h_initial.to(pos.device).unsqueeze(0).expand(n_batch, -1, -1)
            # h = torch.cat([h_t, h_beta], dim=-1) #i.e. node_attr
            """
        else:
            # h = torch.cat([h_t, h_beta], dim=-1)
            h = t

        # h = torch.cat([h, t], dim=-1)

        ###
        # t = t.unsqueeze(-1)
        # beta = beta.unsqueeze(-1)
        # if self.condition_time:
        #     h_t = torch.ones(n_batch, self._n_particles).to(pos.device) * t
        # if self.condition_temperature:
        #     h_beta = torch.ones(n_batch, self._n_particles).to(pos.device) * beta
        # h = h.reshape(n_batch * self._n_particles, self.node_attr_dim)
        ###

        edge_attr = torch.sum((pos_rs[edges[0]] - pos_rs[edges[1]]) ** 2, dim=1, keepdim=True)
        edge_vec = pos_rs[edges[0]] - pos_rs[edges[1]]

        # run the potentially wrapped representation model
        x, v, pos = self.representation_model(
            h=h,
            pos=pos_rs,
            edge_index=edges,
            edge_attr=edge_attr,
            edge_vec=edge_vec,
            # z=z,
        )

        # latent representation
        _, v = self.output_model.pre_reduce(x, v, pos)
        v -= pos_rs
        v = v.view(n_batch, self._n_particles, self._n_dimension)
        v = remove_mean(v, self._n_particles, self._n_dimension)
        return v.view(n_batch, self._n_particles * self._n_dimension)
