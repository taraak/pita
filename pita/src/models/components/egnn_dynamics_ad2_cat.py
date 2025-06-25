import math

import mdtraj as md
import numpy as np
import torch
import torch.nn as nn
from src.models.components.egnn import EGNN
from src.models.components.utils import remove_mean


class EGNN_dynamics_AD2_cat(nn.Module):
    def __init__(
        self,
        n_particles,
        n_dimensions,
        hidden_nf=64,
        act_fn=torch.nn.SiLU(),
        n_layers=5,  # changed to match AD2_classical_train_tgb_full.py
        recurrent=True,
        attention=True,  # changed to match AD2_classical_train_tgb_full.py
        tanh=True,  # changed to match AD2_classical_train_tgb_full.py
        atom_encoding_filename: str = "atom_types_ecoding.npy",
        data_dir="data/alanine",
        pdb_filename="",
        agg="sum",
        M=128,
        condition_beta=False,
    ):
        super().__init__()
        self._n_particles = n_particles
        self._n_dimensions = n_dimensions
        if n_particles >= 53 and n_particles != 55 and n_particles != 13:
            self.data_dir = data_dir
            self.atom_types_encoding = np.load(
                f"{self.data_dir}/{atom_encoding_filename}", allow_pickle=True
            ).item()
            self.pdb_path = f"{self.data_dir}/{pdb_filename}"
            self.topology = md.load_topology(self.pdb_path)
        # Initial one hot encoding of the different element types
        self.h_initial = self.get_h_initial()
        self.condition_beta = condition_beta

        h_size = self.h_initial.size(1)
        h_size += 1

        if self.condition_beta:
            h_size += 1

        self.egnn = EGNN(
            in_node_nf=h_size,
            in_edge_nf=1,
            hidden_nf=hidden_nf,
            act_fn=act_fn,
            n_layers=n_layers,
            recurrent=recurrent,
            attention=attention,
            tanh=tanh,
            agg=agg,
        )

        self.edges = self._create_edges()
        self._edges_dict = {}
        # Count function calls
        self.counter = 0
        self.M = M

    def get_h_initial(self):
        if self._n_particles == 22:
            atom_types = np.arange(22)
            atom_types[[0, 2, 3]] = 2
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
        n_encodings = 78
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

    def forward(self, t, xs, beta):
        x = xs
        t = t.view(-1, 1)

        if t.numel() == 1:
            t = t.repeat(x.shape[0], 1)

        n_batch = x.shape[0]
        edges = self._cast_edges2batch(self.edges, n_batch, self._n_particles, device=x.device)
        edges = [edges[0], edges[1]]

        # Changed by Leon
        x = x.reshape(n_batch * self._n_particles, self._n_dimensions).clone()
        h = self.h_initial.to(x.device).reshape(1, -1)
        h = h.repeat(n_batch, 1)
        h = h.reshape(n_batch * self._n_particles, -1)

        if t.shape != (n_batch, 1):
            t = t.repeat(n_batch)
        t = t.repeat(1, self._n_particles)
        t = t.reshape(n_batch * self._n_particles, 1)

        if self.condition_beta:
            beta = beta.view(-1, 1)
            beta = beta.repeat(1, self._n_particles)
            beta = beta.reshape(n_batch * self._n_particles, 1)
            t = torch.cat([t, beta], dim=-1)

        h = torch.cat([h, t], dim=-1)
        edge_attr = torch.sum((x[edges[0]] - x[edges[1]]) ** 2, dim=1, keepdim=True)
        _, x_final = self.egnn(h, x, edges, edge_attr=edge_attr)
        vel = x_final - x

        vel = vel.view(n_batch, self._n_particles, self._n_dimensions)
        vel = remove_mean(vel)
        self.counter += 1
        return vel.view(n_batch, self._n_particles * self._n_dimensions)

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
