import numpy as np
import torch
import pandas as pd
import torch.distributions as D

import abc
import itertools
import torch
from torch.utils.data import Dataset

from .base_energy_function import BaseEnergyFunction


_LGCP_DIM = 1600

# pylint: disable=invalid-name, too-many-instance-attributes
def read_points(file_path):
    df = pd.read_csv(file_path)
    x_pos, y_pos = np.array(df["data_x"]), np.array(df["data_y"])
    pos = np.vstack([x_pos, y_pos]).T  # (B, 2)
    return pos


def get_bin_counts(array_in, num_bins_per_dim):
    scaled_array = array_in * num_bins_per_dim
    counts = np.zeros((num_bins_per_dim, num_bins_per_dim))
    for elem in scaled_array:
        flt_row, col_row = np.floor(elem)
        row = int(flt_row)
        col = int(col_row)
        # Deal with the case where the point lies exactly on upper/rightmost edge.
        if row == num_bins_per_dim:
            row -= 1
        if col == num_bins_per_dim:
            col -= 1
        counts[row, col] += 1
    return counts


def get_bin_vals(num_bins: int):
    grid_indices = np.arange(num_bins)
    bin_vals = np.array(
        [np.array(elem) for elem in itertools.product(grid_indices, grid_indices)]
    )

    return bin_vals


def th_batch_kernel_fn(x, y, signal_variance, num_grid_per_dim, raw_length_scale):
    x = x.view(-1, 1, x.shape[-1])  # Bx1xL
    y = y.view(1, -1, x.shape[-1])  # 1xBxL
    dist = torch.norm(x - y, dim=2) / (num_grid_per_dim * raw_length_scale)
    return signal_variance * torch.exp(-dist)  # BxB


def get_latents_from_white(white, const_mean, cholesky_gram):
    """
    white: (B,D)
    const_mean: scalar
    cholesky_gram: (D,D)
    """
    return torch.einsum("ij,bj->bi", cholesky_gram, white) + const_mean


def get_white_from_latents(latents, const_mean, cholesky_gram):
    """
    latents: (B,D)
    const_mean: scalar
    cholesky_gram: (D,D)
    """
    white = torch.triangular_solve(
        latents.unsqueeze(-1) - const_mean, cholesky_gram, upper=False
    ).solution  # (B,D, 1)
    return white.squeeze(dim=-1)


def poisson_process_log_likelihood(latent_function, bin_area, flat_bin_counts):
    """
    latent_function: (B,D)
    bin_area: Scalar
    flat_bin_counts: (D)
    """
    first_term = latent_function * flat_bin_counts.unsqueeze(0)  # (B,D)
    second_term = -bin_area * torch.exp(latent_function)
    return torch.sum(first_term + second_term, dim=1)  # (B,)


class BaseSet(abc.ABC, Dataset):
    def __init__(self, len_data=-2333):
        self.num_sample = len_data
        self.data = None
        self.data_ndim = None

    def gt_logz(self):
        raise NotImplementedError

    def energy(self, x):
        return

    def unnorm_pdf(self, x):
        return torch.exp(-self.energy(x))

    # hmt stands for hamiltonian
    def hmt_energy(self, x):
        dim = x.shape[-1]
        x, v = torch.split(x, dim // 2, dim=-1)
        neg_log_p_x = self.sample_energy_fn(x)
        neg_log_p_v = nll_unit_gaussian(v)
        return neg_log_p_x + neg_log_p_v

    @property
    def ndim(self):
        return self.data_ndim

    def sample(self, batch_size):
        del batch_size
        raise NotImplementedError

    def score(self, x):
        with torch.no_grad():
            copy_x = x.detach().clone()
            copy_x.requires_grad = True
            with torch.enable_grad():
                self.energy(copy_x).sum().backward()
                lgv_data = copy_x.grad.data
            return lgv_data

    def hmt_score(self, x):
        with torch.no_grad():
            copy_x = x.detach().clone()
            copy_x.requires_grad = True
            with torch.enable_grad():
                self.hmt_energy(copy_x).sum().backward()
                lgv_data = copy_x.grad.data
            return lgv_data

    def ksd(self, points):
        with torch.no_grad():
            cur_ksd = self.gt_ksd()
        return self.worker(points) - cur_ksd

    def gt_ksd(self):
        if self._gt_ksd is None:
            with torch.no_grad():
                self._gt_ksd = self.worker(
                    self.sample(5000).view(5000, -1), adjust_beta=True
                )
        return self._gt_ksd


class Cox:
    def __init__(self, file_name, num_bins_per_dim, use_whitened):
        self.use_whitened = use_whitened
        self._num_latents = num_bins_per_dim ** 2
        self._num_grid_per_dim = num_bins_per_dim
        bin_counts = np.array(get_bin_counts(read_points(file_name), num_bins_per_dim))

        self._flat_bin_counts = bin_counts.flatten()

        self._poisson_a = 1.0 / self._num_latents

        self._signal_variance = 1.91
        self._beta = 1.0 / 33

        # torch
        self._bin_vals = torch.from_numpy(get_bin_vals(num_bins_per_dim)).cuda().float()

        short_kernel_func = lambda x, y: th_batch_kernel_fn(
            x, y, self._signal_variance, num_bins_per_dim, self._beta
        )
        self._gram_matrix = short_kernel_func(self._bin_vals, self._bin_vals)
        self._cholesky_gram = torch.linalg.cholesky(self._gram_matrix)
        self._white_gaussian_log_normalizer = (
            -0.5 * self._num_latents * np.log(2.0 * np.pi)
        )  # float
        half_log_det_gram = torch.diag(self._cholesky_gram).abs().log().sum()
        self._unwhitened_gaussian_log_normalizer = (
            -0.5 * self._num_latents * np.log(2.0 * np.pi) - half_log_det_gram
        )  # tensor scalar

        self._mu_zero = np.log(126.0) - 0.5 * self._signal_variance  # tensor scalar

        self._flat_bin_counts = torch.from_numpy(self._flat_bin_counts).cuda().float()

        if use_whitened:
            self.evaluate_log_density = self.whitened_posterior_log_density
        else:
            self.evaluate_log_density = self.unwhitened_posterior_log_density

    def whitened_posterior_log_density(self, white):
        # B, _ = white.shape
        quadratic_term = -0.5 * torch.sum(white ** 2, dim=1)  # (B,)
        prior_log_density = self._white_gaussian_log_normalizer + quadratic_term  # (B,)
        latent_function = get_latents_from_white(
            white, self._mu_zero, self._cholesky_gram
        )  # (B,D)
        log_likelihood = poisson_process_log_likelihood(
            latent_function, self._poisson_a, self._flat_bin_counts
        )  # (B,)
        return prior_log_density + log_likelihood  # (B,)

    def unwhitened_posterior_log_density(self, latents):
        white = get_white_from_latents(
            latents, self._mu_zero, self._cholesky_gram
        )  # (B,D)
        prior_log_density = (
            -0.5 * torch.sum(white * white, dim=1)
            + self._unwhitened_gaussian_log_normalizer
        )  # (B,)
        log_likelihood = poisson_process_log_likelihood(
            latents, self._poisson_a, self._flat_bin_counts
        )  # (B,)
        return prior_log_density + log_likelihood


class CoxDist(BaseSet):
    def __init__(self, csv_path, len_data, dim):
        self.cox = Cox(csv_path, 40, use_whitened=False)

        super().__init__(len_data)
        self.data = torch.ones(dim, dtype=float).cuda()  # pylint: disable= not-callable
        self.data_ndim = dim

    def log_pdf(self, x):
        return self.cox.evaluate_log_density(x)

    def get_gt_disc(self, x):
        return -self.cox.evaluate_log_density(x)


class LogGaussianCoxProcessEnergy(BaseEnergyFunction, CoxDist):
    def __init__(
        self,
        dimensionality: int,
        data_normalization_factor: int,
        data_path: str ='../data/df_pines.csv',
    ):
        CoxDist.__init__(self, data_path, -1, _LGCP_DIM)
        BaseEnergyFunction.__init__(self, _LGCP_DIM)

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        return self.log_pdf(samples)
