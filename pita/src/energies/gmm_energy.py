# import sys
# import os
# sys.path.append(os.path.abspath('../'))

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.loggers import WandbLogger
from src.energies.base_energy_function import BaseEnergyFunction
from src.utils.logging_utils import fig_to_image

from fab.fab.target_distributions import gmm
from fab.fab.utils.plotting import plot_contours, plot_marginal_pair


class GMM(BaseEnergyFunction):
    def __init__(
        self,
        dimensionality=2,
        n_mixes=40,
        loc_scaling=40,
        log_var_scaling=1.0,
        mean=None,
        scale=None,
        cat_probs=None,
        device="cpu",
        true_expectation_estimation_n_samples=int(1e5),
        plotting_buffer_sample_size=512,
        plot_samples_epoch_period=5,
        should_unnormalize=False,
        data_normalization_factor=50,
        train_set_size=100000,
        test_set_size=10000,
        val_set_size=10000,
        temperature=1.0,
    ):
        use_gpu = device != "cpu"
        torch.manual_seed(0)  # seed of 0 for GMM problem
        self.gmm = gmm.GMM(
            dim=dimensionality,
            n_mixes=n_mixes,
            mean=mean,
            scale=scale,
            cat_probs=cat_probs,
            loc_scaling=loc_scaling,
            log_var_scaling=log_var_scaling,
            use_gpu=use_gpu,
            true_expectation_estimation_n_samples=true_expectation_estimation_n_samples,
        )

        self.curr_epoch = 0
        self.device = device
        self.plotting_buffer_sample_size = plotting_buffer_sample_size
        self.plot_samples_epoch_period = plot_samples_epoch_period

        self.should_unnormalize = should_unnormalize
        self.data_normalization_factor = data_normalization_factor
        self.train_set_size = train_set_size
        self.test_set_size = test_set_size
        self.val_set_size = val_set_size

        self.temperature = temperature

        self.name = "gmm"

        self.normalization_min = -data_normalization_factor
        self.normalization_max = data_normalization_factor

        super().__init__(
            dimensionality=dimensionality,
            normalization_min=-data_normalization_factor,
            normalization_max=data_normalization_factor,
        )

    def setup_test_set(self):
        test_sample = self.gmm.sample((self.test_set_size,))
        return self.normalize(test_sample)

    def setup_train_set(self):
        train_samples = self.gmm.sample((self.train_set_size,))
        return self.normalize(train_samples)

    def setup_val_set(self):
        val_samples = self.gmm.sample((self.val_set_size,))
        return self.normalize(val_samples)

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        if self.should_unnormalize:
            samples = self.unnormalize(samples)
        return self.gmm.log_prob(samples) / self.temperature

    @property
    def dimensionality(self):
        return 2

    def log_on_start(
        self,
        samples: torch.Tensor,
        wandb_logger: WandbLogger,
        prefix: str = "",
    ):
        if samples is None:
            return
        else:
            if self.should_unnormalize:
                samples = self.unnormalize(samples)
            samples_fig = self.get_dataset_fig(samples)
            wandb_logger.log_image(f"{prefix}initial_samples", [samples_fig])

    def log_on_epoch_end(
        self,
        latest_samples: torch.Tensor,
        latest_energies: torch.Tensor,
        wandb_logger: WandbLogger,
        unprioritized_buffer_samples=None,
        cfm_samples=None,
        replay_buffer=None,
        prefix: str = "",
    ) -> None:
        if latest_samples is None:
            return

        if wandb_logger is None:
            return

        if len(prefix) > 0 and prefix[-1] != "/":
            prefix += "/"

        if self.curr_epoch % self.plot_samples_epoch_period == 0:
            if self.should_unnormalize:
                # Don't unnormalize CFM samples since they're in the
                # unnormalized space
                latest_samples = self.unnormalize(latest_samples)

            samples_fig = self.get_dataset_fig(latest_samples)
            wandb_logger.log_image(f"{prefix}generated_samples", [samples_fig])

            if cfm_samples is not None:
                cfm_samples_fig = self.get_dataset_fig(cfm_samples)
                wandb_logger.log_image(f"{prefix}cfm_generated_samples", [cfm_samples_fig])

            # if unprioritized_buffer_samples is not None:
            #     buffer_samples, _, _ = replay_buffer.sample(self.plotting_buffer_sample_size)

            #     if self.should_unnormalize:
            #         # Don't unnormalize CFM samples since they're in the
            #         # unnormalized space
            #         buffer_samples = self.unnormalize(buffer_samples)
            #         latest_samples = self.unnormalize(latest_samples)

            #         unprioritized_buffer_samples = self.unnormalize(unprioritized_buffer_samples)

            #     samples_fig = self.get_dataset_fig(buffer_samples, latest_samples)

            #     wandb_logger.log_image(f"{prefix}generated_samples", [samples_fig])

            # if latest_samples is not None:
            #     fig, ax = plt.subplots()
            #     ax.scatter(*latest_samples.detach().cpu().T)

            #     wandb_logger.log_image(f"{prefix}generated_samples_scatter", [fig_to_image(fig)])
            #     self.get_single_dataset_fig(latest_samples, "dem_generated_samples")

        self.curr_epoch += 1

    def log_samples(
        self,
        samples: torch.Tensor,
        wandb_logger: WandbLogger,
        name: str = "",
        should_unnormalize: bool = False,
    ) -> None:
        if wandb_logger is None:
            return

        if self.should_unnormalize and should_unnormalize:
            samples = self.unnormalize(samples)
        samples_fig = self.get_single_dataset_fig(samples, name)
        wandb_logger.log_image(f"{name}", [samples_fig])

    def get_single_dataset_fig(self, samples, name, plotting_bounds=(-1.4 * 40, 1.4 * 40)):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        self.gmm.to("cpu")
        plot_contours(
            self.gmm.log_prob,
            bounds=plotting_bounds,
            ax=ax,
            n_contour_levels=50,
            grid_width_n_points=200,
            temperature=self.temperature,
        )

        plot_marginal_pair(samples, ax=ax, bounds=plotting_bounds)
        ax.set_title(f"{name}")

        self.gmm.to(self.device)

        return fig_to_image(fig)

    def get_dataset_fig(
        self,
        samples,
        gen_samples=None,
        plotting_bounds=(-1.4 * 40, 1.4 * 40),
        color="blue",
        cmap=None,
        title=None,
        is_display_fig=False,
    ):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        self.gmm.to("cpu")
        plot_contours(
            self.gmm.log_prob,
            bounds=plotting_bounds,
            ax=axs[0],
            n_contour_levels=50,
            grid_width_n_points=200,
            temperature=self.temperature,
        )

        # plot dataset samples
        plot_marginal_pair(samples, ax=axs[0], bounds=plotting_bounds, color=color, cmap=cmap)
        if title is not None:
            axs[0].set_title(title)
        else:
            axs[0].set_title("Buffer")

        if gen_samples is not None:
            plot_contours(
                self.gmm.log_prob,
                bounds=plotting_bounds,
                ax=axs[1],
                n_contour_levels=50,
                grid_width_n_points=200,
            )
            # plot generated samples
            plot_marginal_pair(
                gen_samples, ax=axs[1], bounds=plotting_bounds, color=color, cmap=cmap
            )
            axs[1].set_title("Generated samples")

        # delete subplot
        else:
            fig.delaxes(axs[1])

        self.gmm.to(self.device)

        if is_display_fig:
            return fig
        # fig.canvas.draw()
        # return PIL.Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        return fig_to_image(fig)


def product_of_gaussians(mu1, sigma1, mu2, sigma2, log_weights):
    var1 = sigma1**2
    var2 = sigma2**2

    denom = var1 + var2
    mu_prod = (mu1 * var2 + mu2 * var1) / denom
    var_prod = (var1 * var2) / denom
    std_prod = var_prod**0.5

    diff = mu1 - mu2

    log_weights = (
        log_weights
        - 0.5 * torch.log(2 * np.pi * torch.prod(denom))
        + torch.sum(-(diff**2) / (2 * denom), dim=-1)
    )

    return mu_prod, std_prod, log_weights


def gmm_product(gmm1, gmm2):
    means_1 = gmm1.locs
    scale_trils_1 = gmm1.scale_trils
    weights_1 = gmm1.cat_probs

    means_2 = gmm2.locs
    scale_trils_2 = gmm2.scale_trils
    weights_2 = gmm2.cat_probs

    K_1 = means_1.shape[0]
    K_2 = means_2.shape[0]

    new_weights = []
    new_means = []
    new_stds = []

    for i in range(K_1):
        for j in range(K_2):
            mu1, sigma1 = means_1[i], torch.diagonal(scale_trils_1[i], dim1=-2, dim2=-1)
            mu2, sigma2 = means_2[j], torch.diagonal(scale_trils_2[i], dim1=-2, dim2=-1)
            log_weights1, log_weights2 = weights_1[i], weights_2[j]

            # Product of two Gaussians
            mu_prod, std_prod, z = product_of_gaussians(
                mu1, sigma1, mu2, sigma2, log_weights1 + log_weights2
            )

            # New weight
            new_weights.append(z)
            new_means.append(mu_prod)
            new_stds.append(std_prod)

    # Stack results into tensors
    device = gmm1.device
    new_weights = torch.stack(new_weights).to(device)
    new_means = torch.stack(new_means).to(device)
    new_stds = torch.stack(new_stds).to(device)

    # drop modes with small logprob
    mask = torch.softmax(new_weights, dim=-1) > 1e-4
    new_weights = new_weights[mask]
    new_means = new_means[mask]
    new_stds = new_stds[mask]

    product_gmm = gmm.GMM(
        dim=gmm1.dim,
        n_mixes=new_weights.shape[0],
        mean=new_means,
        scale=new_stds,
        cat_probs=new_weights,
        loc_scaling=1.0,
        log_var_scaling=1.0,
        use_gpu=True,
        true_expectation_estimation_n_samples=int(1e5),
    )
    return product_gmm


class GMMTempWrapper(GMM):
    def __init__(self, gmm, beta):
        super().__init__(
            dimensionality=gmm.dimensionality,
            n_mixes=gmm.gmm.n_mixes,
            loc_scaling=1.0,
            log_var_scaling=1.0,
            mean=gmm.gmm.locs,
            scale=torch.diagonal(gmm.gmm.scale_trils, dim1=-2, dim2=-1),
            cat_probs=gmm.gmm.cat_probs,
            device=gmm.device,
            should_unnormalize=gmm.should_unnormalize,
        )

        g_prod = gmm.gmm
        for _ in range(beta - 1):
            g_prod = gmm_product(self.gmm, g_prod)
        self.gmm = g_prod

        self._test_set = self.setup_test_set()
        self._val_set = self.setup_val_set()

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        return super().__call__(samples)
