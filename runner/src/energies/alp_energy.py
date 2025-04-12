import logging
import os
from typing import Optional

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import openmm
import torch
import torchvision
from bgflow import OpenMMBridge, OpenMMEnergy
from lightning.pytorch.loggers import WandbLogger
from matplotlib.colors import LogNorm
from openmm import app
from src.energies.base_energy_function import BaseEnergyFunction
from src.energies.base_molecule_energy_function import BaseMoleculeEnergy
from src.models.components.distribution_distances import (
    compute_distribution_distances_with_prefix,
)
from src.models.components.optimal_transport import torus_wasserstein
from src.models.components.energy_utils import (
    check_symmetry_change,
    compute_chirality_sign,
    find_chirality_centers,
)
from src.utils.data_utils import remove_mean

logger = logging.getLogger(__name__)


class ALPEnergy(BaseMoleculeEnergy):
    def __init__(
        self,
        data_path: str,
        pdb_filename: str,
        atom_encoding_filename: str = "atom_types_ecoding.npy",
        dimensionality: int=99,
        n_particles: int=33,
        spatial_dim: int=3,
        device: str="cpu",
        plot_samples_epoch_period: int=5,
        plotting_buffer_sample_size: int=512,
        data_normalization_factor: float=1.0,
        is_molecule: bool=True,
        temperature: float=1.0,
        should_normalize: bool=True,
    ):
        super().__init__(
            dimensionality=dimensionality,
            n_particles=n_particles,
            spatial_dim=spatial_dim,
            data_path=data_path,
            data_name="AL",
            device=device,
            temperature=temperature,
            should_normalize=should_normalize,
            data_normalization_factor=data_normalization_factor,
            plot_samples_epoch_period=plot_samples_epoch_period,
            plotting_buffer_sample_size=plotting_buffer_sample_size,
            is_molecule=is_molecule,
        )
        
        self.adj_list = None
        self.atom_types = None
        self.atom_encoding_filename = atom_encoding_filename
        self.atom_types_encoding = np.load(data_path + f"/{atom_encoding_filename}", allow_pickle=True
        ).item()

        self.pdb_path =  f"{pdb_filename}"
        logger.info(f"Loading pdb file from {self.pdb_path}")
        self.topology = md.load_topology(self.pdb_path)
        self.pdb = app.PDBFile(self.pdb_path)

        self.adj_list, self.atom_types = self.compute_adj_list_and_atom_types()


        forcefield = app.ForceField("amber14-all.xml", "implicit/obc1.xml")

        system = forcefield.createSystem(
            self.pdb.topology,
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=2.0 * openmm.unit.nanometer,
            constraints=None,
        )
        integrator = openmm.LangevinMiddleIntegrator(
            self.temperature * openmm.unit.kelvin,
            0.3 / openmm.unit.picosecond,
            1.0 * openmm.unit.femtosecond,
        )
        self.openmm_energy = OpenMMEnergy(
            bridge=OpenMMBridge(system, integrator, platform_name="CUDA")
        )


    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        if self.should_normalize:
            samples = self.unnormalize(samples)
        return -self.openmm_energy.energy(samples).squeeze(-1)

    def setup_test_set(self):
        data = np.load(self.data_path_test, allow_pickle=True)
        if self.should_normalize:
            data = self.normalize(data)
        else:
            data = remove_mean(data, self.n_particles, self.n_spatial_dim)
        data = remove_mean(data, self.n_particles, self.n_spatial_dim)
        data = torch.tensor(data, device=self.device)
        return data

    def setup_val_set(self):
        if self.data_path_val is None:
            raise ValueError("Data path for validation data is not provided")
        data = np.load(self.data_path_val, allow_pickle=True)
        if self.should_normalize:
            data = self.normalize(data)
        else:
            data = remove_mean(data, self.n_particles, self.n_spatial_dim)
        data = torch.tensor(data, device=self.device)
        return data

    def setup_train_set(self):
        if self.data_path_train is None:
            raise ValueError("Data path for training data is not provided")
        data = np.load(self.data_path_val, allow_pickle=True)
        if self.should_normalize:
            data = self.normalize(data)
        else:
            data = remove_mean(data, self.n_particles, self.n_spatial_dim)
        data = torch.tensor(data, device=self.device)
        return data

    def get_dataset_fig(
        self,
        samples,
        log_p_samples: torch.Tensor,
        samples_jarzynski: torch.Tensor = None,
        use_com_energy: bool = False,
        min_energy=-20,
        max_energy=80,
        ylim=(0, 0.1),
    ):
        if self.n_particles == 63:
            min_energy = -130
            max_energy = -30
            ylim = (0, 0.1)
        if self.n_particles == 53:
            min_energy = -150
            max_energy = -60
            ylim = (0, 0.1)
        if self.n_particles == 42:
            min_energy = -20
            max_energy = 80
            ylim = (0, 0.1)
        if self.n_particles == 33:
            min_energy = -200
            max_energy = -100
            ylim = (0, 0.1)
        return super().get_dataset_fig(
            samples,
            log_p_samples,
            samples_jarzynski,
            use_com_energy,
            min_energy,
            max_energy,
            ylim=ylim,
        )

    def compute_adj_list_and_atom_types(self):
        atom_dict = {"C": 0, "H": 1, "N": 2, "O": 3, "S": 4}
        atom_types = []
        for atom_name in self.topology.atoms:
            atom_types.append(atom_name.name[0])
        atom_types = torch.from_numpy(np.array([atom_dict[atom_type] for atom_type in atom_types]))
        adj_list = torch.from_numpy(
            np.array([(b.atom1.index, b.atom2.index) for b in self.topology.bonds], dtype=np.int32)
        )
        return adj_list, atom_types

    def log_on_epoch_end(
        self,
        samples,
        log_p_samples: torch.Tensor,
        num_eval_samples: int = 5000,
        use_com_energy: bool = False,
        loggers=None,
        prefix: str = "",
    ) -> None:
        wandb_logger = self.get_wandb_logger(loggers)
        super().log_on_epoch_end(
            samples,
            log_p_samples,
            samples_jarzynski,
            use_com_energy=use_com_energy,
            loggers=loggers,
            prefix=prefix,
        )
        logging.info("Base plots done")

        metrics = {}
        if self.should_normalize:
            samples = self.unnormalize(samples).cpu()
        samples_metrics = self.get_ramachandran_metrics(
            samples[:num_eval_samples], prefix=prefix + "generated_samples/rama"
        )
        logging.info("Ramachandran metrics computed (generated samples)")
        metrics.update(samples_metrics)

        if "val" in prefix:
            reference_samples = self.sample_val_set(5000)
        if "test" in prefix:
            reference_samples = self.sample_test_set(5000)
        chirality_centers = find_chirality_centers(self.adj_list, self.atom_types)
        reference_signs = compute_chirality_sign(
            reference_samples.reshape(-1, self.n_particles, self.n_spatial_dim)[[1]], chirality_centers
        )
        symmetry_change = check_symmetry_change(
            samples, chirality_centers, reference_signs
        )
        print("Symmetry change frac:", (symmetry_change).float().mean())
        samples[symmetry_change] *= -1
        correct_symmetry_rate = 1 - symmetry_change.sum() / len(symmetry_change)
        symmetry_change = check_symmetry_change(
            samples, chirality_centers, reference_signs
        )
        samples = samples[~symmetry_change]
        uncorrectable_symmetry_rate = symmetry_change.sum() / len(symmetry_change)
        samples = samples.reshape(-1, self.n_particles * self.n_spatial_dim)

        metrics.update(
            {
                prefix + "/correct_symmetry_rate": correct_symmetry_rate,
                prefix + "/uncorrectable_symmetry_rate": uncorrectable_symmetry_rate,
            }
        )
        self.plot_tica(None, prefix=prefix + "/ground_truth", wandb_logger=wandb_logger)

        logging.info("Ramachandran metrics computed")

        return metrics

    def get_ramachandran_metrics(self, samples, prefix: str = ""):
        x_pred = self.get_phi_psi_vectors(samples)

        if "val" in prefix:
            eval_samples = self.sample_val_set(x_pred.shape[0])
        elif "test" in prefix:
            eval_samples = self.sample_test_set(x_pred.shape[0])
        if self.should_normalize:
            eval_samples = self.unnormalize(eval_samples)
        x_true = self.get_phi_psi_vectors(eval_samples)

        # metrics = compute_distribution_distances_with_prefix(x_true, x_pred, prefix=prefix)
        # metrics[prefix + "/torus_wasserstein"] = torus_wasserstein(x_true, x_pred)
        return metrics

    def get_phi_psi_vectors(self, samples):
        samples = samples.reshape(-1, self.n_particles, self.n_spatial_dim)
        traj_samples = md.Trajectory(samples, topology=self.topology)
        phis = md.compute_phi(traj_samples)[1]
        psis = md.compute_psi(traj_samples)[1]
        x = torch.cat([torch.from_numpy(phis), torch.from_numpy(psis)], dim=1)
        return x

    def plot_ramachandran(self, samples, prefix: str = "", wandb_logger: WandbLogger = None):
        samples = samples.reshape(-1, self.n_particles, self.n_spatial_dim)
        traj_samples = md.Trajectory(samples, topology=self.topology)
        phis = md.compute_phi(traj_samples)[1]
        psis = md.compute_psi(traj_samples)[1]
        for i in range(phis.shape[1]):
            print(f"Plotting Ramachandran {i} out of {phis.shape[1]}")
            phi_tmp = phis[:, i]
            psi_tmp = psis[:, i]
            fig, ax = plt.subplots()
            plot_range = [-np.pi, np.pi]
            h, x_bins, y_bins, im = ax.hist2d(
                phi_tmp,
                psi_tmp,
                100,
                norm=LogNorm(),
                range=[plot_range, plot_range],
                rasterized=True,
            )
            ticks = np.array(
                [np.exp(-6) * h.max(), np.exp(-4.0) * h.max(), np.exp(-2) * h.max(), h.max()]
            )
            ax.set_xlabel(r"$\varphi$", fontsize=45)
            ax.set_ylabel(r"$\psi$", fontsize=45)
            ax.xaxis.set_tick_params(labelsize=25)
            ax.yaxis.set_tick_params(labelsize=25)
            ax.yaxis.set_ticks([])
            cbar = fig.colorbar(im, ticks=ticks)
            cbar.ax.set_yticklabels([6.0, 4.0, 2.0, 0.0], fontsize=25)

            cbar.ax.invert_yaxis()
            cbar.ax.set_ylabel(r"Free energy / $k_B T$", fontsize=35)
            if wandb_logger is not None:
                wandb_logger.log_image(f"{prefix}/ramachandran/{i}", [fig])

            phi_tmp = phis[:, i]
            psi_tmp = psis[:, i]
            fig, ax = plt.subplots()
            plot_range = [-np.pi, np.pi]
            h, x_bins, y_bins, im = ax.hist2d(
                phi_tmp,
                psi_tmp,
                100,
                norm=LogNorm(),
                range=[plot_range, plot_range],
                rasterized=True,
            )
            ax.set_xlabel(r"$\varphi$", fontsize=45)
            ax.set_ylabel(r"$\psi$", fontsize=45)
            ax.xaxis.set_tick_params(labelsize=25)
            ax.yaxis.set_tick_params(labelsize=25)
            ax.yaxis.set_ticks([])
            cbar = fig.colorbar(im)  # , ticks=ticks)
            im.set_clim(vmax=samples.shape[0] // 20)
            cbar.ax.set_ylabel(f"Count, max = {int(h.max())}", fontsize=18)
            if wandb_logger is not None:
                wandb_logger.log_image(f"{prefix}/ramachandran_simple/{i}", [fig])

        return fig
