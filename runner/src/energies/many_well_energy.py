import torch
import matplotlib.pyplot as plt

from typing import Optional
from lightning.pytorch.loggers import WandbLogger
from fab.target_distributions.many_well import ManyWellEnergy
from fab.utils.plotting import plot_contours, plot_marginal_pair

from src.models.components.replay_buffer import ReplayBuffer
from src.energies.base_energy_function import BaseEnergyFunction
from src.utils.logging_utils import fig_to_image


class ManyWell(BaseEnergyFunction):
    def __init__(
        self,
        dimensionality=6,
        device="cpu",
        true_expectation_estimation_n_samples=int(1e5),
        plotting_buffer_sample_size=512,
        plot_samples_epoch_period=5,
        test_set_size=1024,
        train_set_size=100000,
        should_unnormalize=False,
        data_normalization_factor=3,
    ):
        self.many_well = ManyWellEnergy(
            dimensionality, a=-0.5, b=-6, use_gpu=device != "cpu"
        )

        self.curr_epoch = 0
        self.device = device
        self.plotting_buffer_sample_size = plotting_buffer_sample_size
        self.plot_samples_epoch_period = plot_samples_epoch_period

        self.test_set_size = test_set_size

        self.should_unnormalize = should_unnormalize

        super().__init__(
            dimensionality=dimensionality,
            normalization_min=-data_normalization_factor,
            normalization_max=data_normalization_factor,
        )

    def setup_test_set(self):
        return self.many_well.sample((self.test_set_size,))

    def setup_train_set(self):
        return self.many_well.sample((self.train_set_size,))

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        if self.should_unnormalize:
            samples = self.unnormalize(samples)

        return self.many_well.log_prob(samples)

    def log_on_epoch_end(
        self,
        latest_samples: torch.Tensor,
        latest_energies: torch.Tensor,
        unprioritized_buffer_samples: Optional[torch.Tensor],
        cfm_samples: Optional[torch.Tensor],
        replay_buffer: ReplayBuffer,
        wandb_logger: WandbLogger,
        prefix: str = "",
    ) -> None:
        if wandb_logger is None:
            return

        if len(prefix) > 0 and prefix[-1] != "/":
            prefix += "/"

        if self.curr_epoch % self.plot_samples_epoch_period == 0:
            buffer_samples, _, _ = replay_buffer.sample(
                self.plotting_buffer_sample_size
            )

            if self.should_unnormalize:
                buffer_samples = self.unnormalize(buffer_samples)
                latest_samples = self.unnormalize(latest_samples)

                if unprioritized_buffer_samples is not None:
                    unprioritized_buffer_samples = self.unnormalize(
                        unprioritized_buffer_samples
                    )

            samples_fig = self.get_dataset_fig(buffer_samples, latest_samples)

            wandb_logger.log_image(f"{prefix}generated_samples", [samples_fig])

            if unprioritized_buffer_samples is not None:
                cfm_samples_fig = self.get_dataset_fig(
                    unprioritized_buffer_samples, cfm_samples
                )

                wandb_logger.log_image(
                    f"{prefix}cfm_generated_samples", [cfm_samples_fig]
                )

        self.curr_epoch += 1

    def get_dataset_fig(self, samples, gen_samples=None, plotting_bounds=(-3, 3)):
        n_rows = self.dimensionality // 2
        fig, axs = plt.subplots(
            self.dimensionality // 2,
            2,
            sharex=True,
            sharey=True,
            figsize=(10, n_rows * 3),
        )

        self.many_well.to("cpu")
        if n_rows > 1:
            for i in range(n_rows):
                self._plot_dim(
                    fig,
                    axs[i],
                    plotting_bounds,
                    samples,
                    gen_samples,
                    i
                )

        else:
            self._plot_dim(
                fig,
                axs,
                plotting_bounds,
                samples,
                gen_samples,
                0
            )

        ax = axs[0] if n_rows > 1 else axs
        ax[0].set_title("Dataset")
        ax[1].set_title("Generated samples")

        plt.tight_layout()

        self.many_well.to(self.device)
        return fig_to_image(fig)

    def _plot_dim(
        self,
        fig,
        ax,
        plotting_bounds,
        samples,
        gen_samples,
        i
    ):
        plot_contours(
            self.many_well.log_prob_2D,
            bounds=plotting_bounds,
            ax=ax[0],
            n_contour_levels=40,
        )

        # plot buffer samples
        plot_marginal_pair(
            samples,
            ax=ax[0],
            bounds=plotting_bounds,
            marginal_dims=(i * 2, i * 2 + 1),
        )

        ax[0].set_xlabel(f"dim {i*2}")
        ax[0].set_ylabel(f"dim {i*2 + 1}")

        if gen_samples is not None:
            plot_contours(
                self.many_well.log_prob_2D,
                bounds=plotting_bounds,
                ax=ax[1],
                n_contour_levels=40,
            )

            # plot generated samples
            plot_marginal_pair(
                gen_samples,
                ax=ax[1],
                bounds=plotting_bounds,
                marginal_dims=(i * 2, i * 2 + 1),
            )

            ax[1].set_xlabel(f"dim {i*2}")
            ax[1].set_ylabel(f"dim {i*2 + 1}")
        else:
            fig.delaxes(ax[1])
