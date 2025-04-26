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
from src.energies.components.tica import plot_tic01, run_tica, tica_features
from src.models.components.distribution_distances import (
    compute_distribution_distances_with_prefix,
)
from src.models.components.energy_utils import (
    check_symmetry_change,
    compute_chirality_sign,
    find_chirality_centers,
)
from src.models.components.optimal_transport import torus_wasserstein
from src.utils.data_utils import remove_mean

logger = logging.getLogger(__name__)


class ALPEnergy(BaseMoleculeEnergy):
    def __init__(
        self,
        data_path: str,
        pdb_filename: str,
        atom_encoding_filename: str = "atom_types_ecoding.npy",
        dimensionality: int = 99,
        n_particles: int = 33,
        spatial_dim: int = 3,
        device: str = "cpu",
        plot_samples_epoch_period: int = 5,
        plotting_buffer_sample_size: int = 512,
        data_normalization_factor: float = 1.0,
        is_molecule: bool = True,
        temperature: float = 1.0,
        should_normalize: bool = True,
        device_index: int = 0,
        debug_train_on_test: bool = False,
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

        self.debug_train_on_test = debug_train_on_test
        self.adj_list = None
        self.atom_types = None
        self.atom_encoding_filename = atom_encoding_filename
        self.atom_types_encoding = np.load(
            data_path + f"/{atom_encoding_filename}", allow_pickle=True
        ).item()

        self.pdb_path = f"{pdb_filename}"
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

        platform_name = "CUDA"
        platform_properties = (
            dict(Precision="single", DeviceIndex=device_index)
            if platform_name == "CUDA"
            else dict()
        )
        self.openmm_energy = OpenMMEnergy(
            bridge=OpenMMBridge(
                system,
                integrator,
                platform_name=platform_name,
                platform_properties=platform_properties,
            ),
        )

    def __call__(self, samples: torch.Tensor, return_force=False) -> torch.Tensor:
        if self.should_normalize:
            samples = self.unnormalize(samples)
        logprob = -self.openmm_energy.energy(samples).squeeze(-1)
        if return_force:
            force = self.openmm_energy.force(samples)
            return logprob, force
        return logprob

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
        path = self.data_path_train
        if self.debug_train_on_test:
            path = self.data_path_test
        data = np.load(path, allow_pickle=True)
        if self.should_normalize:
            data = self.normalize(data)
        else:
            data = remove_mean(data, self.n_particles, self.n_spatial_dim)
        data = torch.tensor(data, device=self.device)
        return data

    def get_dataset_fig(
        self,
        samples,
        energy_samples: torch.Tensor,
        samples_not_resampled: Optional[torch.Tensor] = None,
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
            energy_samples,
            samples_not_resampled=samples_not_resampled,
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
        latest_samples: torch.Tensor,
        latest_energies: torch.Tensor,
        wandb_logger: WandbLogger,
        latest_samples_not_resampled: Optional[torch.Tensor] = None,
        num_eval_samples: int = 5000,
        prefix: str = "",
    ) -> None:
        super().log_on_epoch_end(
            latest_samples,
            latest_energies,
            wandb_logger=wandb_logger,
            latest_samples_not_resampled=latest_samples_not_resampled,
            prefix=prefix,
        )
        logging.info("Base plots done")

        metrics = {}
        if self.should_normalize:
            latest_samples = self.unnormalize(latest_samples).cpu()
        samples_metrics = self.get_ramachandran_metrics(
            latest_samples[:num_eval_samples], prefix=prefix + "generated_samples/rama"
        )
        try:
            self.plot_ramachandran(
                latest_samples,
                prefix=prefix + "/generated_samples/rama",
                wandb_logger=wandb_logger,
            )
        except ValueError as e:
            logging.error(f"Error in plotting Ramachandran: {e}")
        logging.info("Ramachandran metrics computed (generated samples)")
        metrics.update(samples_metrics)

        if "val" in prefix:
            reference_samples = self.sample_val_set(5000)
        if "test" in prefix:
            reference_samples = self.sample_test_set(5000)
        chirality_centers = find_chirality_centers(self.adj_list, self.atom_types)
        reference_signs = compute_chirality_sign(
            reference_samples.reshape(-1, self.n_particles, self.n_spatial_dim)[[1]],
            chirality_centers,
        )
        symmetry_change = check_symmetry_change(
            latest_samples.reshape(-1, self.n_particles, self.n_spatial_dim),
            chirality_centers,
            reference_signs,
        )
        print("Symmetry change frac:", (symmetry_change).float().mean())
        latest_samples[symmetry_change] *= -1
        correct_symmetry_rate = 1 - symmetry_change.sum() / len(symmetry_change)
        symmetry_change = check_symmetry_change(
            latest_samples.reshape(-1, self.n_particles, self.n_spatial_dim),
            chirality_centers,
            reference_signs,
        )
        latest_samples = latest_samples[~symmetry_change]
        uncorrectable_symmetry_rate = symmetry_change.sum() / len(symmetry_change)
        latest_samples = latest_samples.reshape(-1, self.n_particles * self.n_spatial_dim)

        metrics.update(
            {
                prefix + "/correct_symmetry_rate": correct_symmetry_rate,
                prefix + "/uncorrectable_symmetry_rate": uncorrectable_symmetry_rate,
            }
        )
        self.plot_tica(None, prefix=prefix + "/ground_truth", wandb_logger=wandb_logger)

        return metrics

    def get_ramachandran_metrics(self, samples, prefix: str = ""):
        x_pred = self.get_phi_psi_vectors(samples.cpu())

        if "val" in prefix:
            eval_samples = self.sample_val_set(x_pred.shape[0])
        elif "test" in prefix:
            eval_samples = self.sample_test_set(x_pred.shape[0])
        if self.should_normalize:
            eval_samples = self.unnormalize(eval_samples)
        x_true = self.get_phi_psi_vectors(eval_samples.cpu())

        metrics = compute_distribution_distances_with_prefix(x_true, x_pred, prefix=prefix)
        metrics[prefix + "/torus_wasserstein"] = torus_wasserstein(x_true, x_pred)
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
            ax.set_xlabel(r"$\varphi$", fontsize=25)
            ax.set_ylabel(r"$\psi$", fontsize=25)
            ax.xaxis.set_tick_params(labelsize=25)
            ax.yaxis.set_tick_params(labelsize=25)
            ax.yaxis.set_ticks([])
            cbar = fig.colorbar(im, ticks=ticks)
            cbar.ax.set_yticklabels([6.0, 4.0, 2.0, 0.0], fontsize=25)

            cbar.ax.invert_yaxis()
            cbar.ax.set_ylabel(r"Free energy / $k_B T$", fontsize=25)
            if wandb_logger is not None:
                wandb_logger.log_image(f"{prefix}/ramachandran/{i}", [fig])
        return fig

    def plot_tica(
        self,
        samples=None,
        prefix="",
        wandb_logger=None,
    ):
        lagtime = 10 if self.n_particles == 33 else 100

        test_samples = self.sample_test_set(5000).cpu()
        traj_samples_test = md.Trajectory(
            test_samples.reshape(-1, self.n_particles, self.n_spatial_dim).numpy(),
            topology=self.topology,
        )

        # the tica projection is computed based on reference data
        # the lagtime can be changed in order to get well seperated states
        tica_model = run_tica(traj_samples_test, lagtime=lagtime)

        if samples is None:
            # we can then map other data, e.g. generated with the same transformation
            features = tica_features(traj_samples_test)
        else:
            samples = md.Trajectory(
                samples.reshape(-1, self.n_particles, self.n_spatial_dim).cpu().numpy(),
                topology=self.topology,
            )
            features = tica_features(samples)
        tics = tica_model.transform(features)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax = plot_tic01(ax, tics, f"MD", tics_lims=tics)

        if wandb_logger is not None:
            wandb_logger.log_image(f"{prefix}/tica/plot", [fig])

        return fig
