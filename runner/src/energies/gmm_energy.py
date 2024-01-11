import torch
import PIL
import matplotlib.pyplot as plt

from lightning.pytorch.loggers import WandbLogger
from fab.target_distributions import gmm
from fab.utils.plotting import plot_contours, plot_marginal_pair

from src.models.components.replay_buffer import ReplayBuffer
from src.energies.base_energy_function import BaseEnergyFunction

def fig_to_image(fig):
    fig.canvas.draw()

    return PIL.Image.frombytes(
        'RGB',
        fig.canvas.get_width_height(),
        fig.canvas.tostring_rgb()
    )

class GMM(BaseEnergyFunction):
    def __init__(
        self,
        dimensionality=2,
        n_mixes=40,
        loc_scaling=40,
        log_var_scaling=1.0,
        device="cpu",
        true_expectation_estimation_n_samples=int(1e5),
        plotting_buffer_sample_size=512,
        plot_samples_epoch_period=5
    ):
        use_gpu = device != "cpu"
        torch.manual_seed(0)  # seed of 0 for GMM problem
        self.gmm = gmm.GMM(
            dim=dimensionality,
            n_mixes=n_mixes,
            loc_scaling=loc_scaling,
            log_var_scaling=log_var_scaling,
            use_gpu=use_gpu,
            true_expectation_estimation_n_samples=true_expectation_estimation_n_samples,
        )

        self.curr_epoch = 0
        self.device = device
        self.plotting_buffer_sample_size = plotting_buffer_sample_size
        self.plot_samples_epoch_period = plot_samples_epoch_period

        super().__init__(dimensionality=dimensionality)

    def setup_test_set(self):
        return self.gmm.test_set

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        return self.gmm.log_prob(samples)

    @property
    def dimensionality(self):
        return 2

    def unnormalize(self, x, mins=-50, maxs=50):
        '''
            x : [ -1, 1 ]
        '''
        x = (x + 1) / 2
        return x * (maxs - mins) + mins

    def log_on_epoch_end(
        self,
        latest_samples: torch.Tensor,
        latest_energies: torch.Tensor,
        replay_buffer: ReplayBuffer,
        wandb_logger: WandbLogger,
        prefix: str = ''
    ) -> None:
        if wandb_logger is None:
            return

        if len(prefix) > 0 and prefix[-1] != '/':
            prefix += '/'

        if self.curr_epoch % self.plot_samples_epoch_period == 0:
            buffer_samples, _, _ = replay_buffer.sample(
                self.plotting_buffer_sample_size
            )

            samples_fig = self.get_dataset_fig(
                buffer_samples,
                latest_samples
            )

            wandb_logger.log_image(
                f'{prefix}generated_samples',
                [samples_fig]
            )

            if latest_samples is not None:
                fig, ax = plt.subplots()
                ax.scatter(*latest_samples.detach().cpu().T)

                wandb_logger.log_image(
                    f'{prefix}generated_samples_scatter',
                    [fig_to_image(fig)]
                )

        self.curr_epoch += 1

    def get_dataset_fig(
        self,
        samples,
        gen_samples=None,
        plotting_bounds=(-1.4 * 40, 1.4 * 40)
    ):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        self.gmm.to("cpu")
        plot_contours(
            self.gmm.log_prob,
            bounds=plotting_bounds,
            ax=axs[0],
            n_contour_levels=50,
            grid_width_n_points=200
        )

        # plot dataset samples
        plot_marginal_pair(samples, ax=axs[0], bounds=plotting_bounds)
        axs[0].set_title("Buffer")

        if gen_samples is not None:
            plot_contours(
                self.gmm.log_prob,
                bounds=plotting_bounds,
                ax=axs[1],
                n_contour_levels=50,
                grid_width_n_points=200
            )
            # plot generated samples
            plot_marginal_pair(gen_samples, ax=axs[1], bounds=plotting_bounds)
            axs[1].set_title("Generated samples")

        # delete subplot
        else:
            fig.delaxes(axs[1])

        self.gmm.to(self.device)

        return fig_to_image(fig)
