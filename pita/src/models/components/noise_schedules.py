from abc import ABC, abstractmethod

import numpy as np
import torch


class BaseNoiseSchedule(ABC):
    @abstractmethod
    def g(t):
        # Returns g(t)
        pass

    @abstractmethod
    def h(t):
        # Returns \int_0^t g(t)^2 dt
        pass


class LinearNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, beta):
        self.beta = beta

    def g(self, t):
        return torch.full_like(t, self.beta**0.5)

    def h(self, t):
        return self.beta * t


class QuadraticNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, beta):
        self.beta = beta

    def g(self, t):
        return torch.sqrt(self.beta * 2 * t)

    def h(self, t):
        return self.beta * t**2


class PowerNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, beta, power):
        self.beta = beta
        self.power = power

    def g(self, t):
        return torch.sqrt(self.beta * self.power * (t ** (self.power - 1)))

    def h(self, t):
        return self.beta * (t**self.power)


class SubLinearNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, beta):
        self.beta = beta

    def g(self, t):
        return torch.sqrt(self.beta * 0.5 * 1 / (t**0.5 + 1e-3))

    def h(self, t):
        return self.beta * t**0.5


class GeometricNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, sigma_min, sigma_max):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_diff = self.sigma_max / self.sigma_min

    def g(self, t):
        # Let sigma_d = sigma_max / sigma_min
        # Then g(t) = sigma_min * sigma_d^t * sqrt{2 * log(sigma_d)}
        # See Eq 192 in https://arxiv.org/pdf/2206.00364.pdf
        return self.sigma_min * (self.sigma_diff**t) * ((2 * np.log(self.sigma_diff)) ** 0.5)

    def h(self, t):
        # Let sigma_d = sigma_max / sigma_min
        # Then h(t) = \int_0^t g(z)^2 dz = sigma_min * sqrt{sigma_d^{2t} - 1}
        # see Eq 199 in https://arxiv.org/pdf/2206.00364.pdf
        return (self.sigma_min * (((self.sigma_diff ** (2 * t)) - 1) ** 0.5)) ** 2

    def sample_ln_sigma(self, num_samples, device):
        # Sample from a uniform distribution U[ln(sigma_min), ln(sigma_max)]
        ln_sigmat = torch.rand(num_samples, device=device) * (
            np.log(self.sigma_max) - np.log(self.sigma_min)
        ) + np.log(self.sigma_min)
        return ln_sigmat

    def get_ln_sigmat_bins(self, num_bins):
        bin_edges = np.linspace(
            np.log(self.sigma_min),
            np.log(self.sigma_max),
            num_bins + 1,
        )
        return bin_edges


class ElucidatingNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, sigma_min, sigma_max, rho, P_mean=-1.2, P_std=1.2):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.term1 = self.sigma_max ** (1 / self.rho)
        self.term2 = self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        self.P_mean = P_mean
        self.P_std = P_std

    def g(self, t):
        # take derivative of h(t) with respect to t with automatic differentiation
        return (
            -2 * self.rho * (self.term1 + (1 - t) * self.term2) ** (2 * self.rho - 1) * self.term2
        ) ** 0.5

    def h(self, t):
        return (self.term1 + (1 - t) * self.term2) ** (2 * self.rho)

    def t(self, ht):
        # Inverse of h(t)
        return 1 - ((ht ** (1 / (2 * self.rho)) - self.term1) / self.term2)

    def dh_dt(self, t):
        # Derivative of h(t) with respect to t
        return (
            -2 * self.rho * self.term2 * (self.term1 + (1 - t) * self.term2) ** (2 * self.rho - 1)
        )

    def sample_ln_sigma(self, num_samples, device):
        # Sample from a normal distribution N(P_mean, P_std)
        ln_sigmat = torch.randn(num_samples, device=device) * self.P_std + self.P_mean
        return ln_sigmat

    def get_ln_sigmat_bins(self, num_bins):
        bin_edges = np.linspace(
            self.P_mean - 2 * self.P_std,
            self.P_mean + 2 * self.P_std,
            num_bins + 1,
        )
        return bin_edges
