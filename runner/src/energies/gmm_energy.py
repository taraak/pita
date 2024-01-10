import torch
from fab.target_distributions import gmm
from src.energies.base_energy_function import BaseEnergyFunction


class GMM(BaseEnergyFunction):
    def __init__(
        self,
        dimensionality=2,
        n_mixes=40,
        loc_scaling=40,
        log_var_scaling=1.0,
        device="cpu",
        true_expectation_estimation_n_samples=int(1e5),
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
        super().__init__(dimensionality=dimensionality)

    def setup_test_set(self):
        return self.gmm.test_set

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        return self.gmm.log_prob(samples)
