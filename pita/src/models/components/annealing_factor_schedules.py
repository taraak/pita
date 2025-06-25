import math
from abc import ABC, abstractmethod

import numpy as np
import torch


class BaseAnnealingFactorSchedule(ABC):
    @abstractmethod
    def gamma(t):
        # Returns inverse temperature beta(t)
        pass

    @abstractmethod
    def dgamma_dt(t):
        # Returns derivative of beta(t) with respect to t
        pass


class ConstantAnnealingFactorSchedule(BaseAnnealingFactorSchedule):
    def __init__(self, annealing_factor):
        self.annealing_factor = annealing_factor

    def gamma(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return torch.ones_like(t) * self.annealing_factor

    def dgamma_dt(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return torch.zeros_like(t)


class LinearAnnealingFactorSchedule(BaseAnnealingFactorSchedule):
    def __init__(self, annealing_factor, annealing_factor_start, t_start=1.0, t_end=0.0):
        self.annealing_factor_start = annealing_factor_start
        self.annealing_factor = annealing_factor

        self.t_start = t_start
        self.t_end = t_end

    def gamma(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)

        # Linear interpolation for reversed time
        slope = (self.annealing_factor - self.annealing_factor_start) / (self.t_end - self.t_start)
        linear_part = slope * (t - self.t_start) + self.annealing_factor_start

        gamma_val = torch.where(
            t > self.t_start,
            self.annealing_factor_start,
            torch.where(t < self.t_end, self.annealing_factor, linear_part),
        )
        return gamma_val

    def dgamma_dt(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)

        slope = (self.annealing_factor - self.annealing_factor_start) / (self.t_end - self.t_start)

        dgamma = torch.where(
            t > self.t_start,
            torch.tensor(0.0, dtype=t.dtype),
            torch.where(t < self.t_end, torch.tensor(0.0, dtype=t.dtype), slope),
        )
        return dgamma


class SigmoidAnnealingFactorSchedule(BaseAnnealingFactorSchedule):
    def __init__(
        self, annealing_factor, annealing_factor_start, t_start=1.0, t_end=0.0, sharpness=10.0
    ):
        self.annealing_factor_start = annealing_factor_start
        self.annealing_factor = annealing_factor
        self.t_start = t_start
        self.t_end = t_end
        self.center = (t_start + t_end) / 2
        self.width = t_start - t_end

        self.sharpness = sharpness

    def gamma(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)

        # Smooth sigmoid transition
        x = (self.center - t) / self.width
        exp_term = torch.exp(-self.sharpness * x)
        smooth = 1 / (1 + exp_term)
        return (
            self.annealing_factor_start
            + (self.annealing_factor - self.annealing_factor_start) * smooth
        )

    def dgamma_dt(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)

        x = (self.center - t) / self.width
        exp_term = torch.exp(-self.sharpness * x)
        smooth = 1 / (1 + exp_term)

        # Derivative of sigmoid w.r.t t
        d_smooth_dt = (self.sharpness / self.width) * smooth * (1 - smooth)

        return (self.annealing_factor - self.annealing_factor_start) * d_smooth_dt
