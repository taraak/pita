from abc import ABC, abstractmethod
import torch
import numpy as np
import torch


class BaseInverseTempSchedule(ABC):
    @abstractmethod
    def beta(t):
        # Returns inverse temperature beta(t)
        pass

    @abstractmethod
    def dbeta_dt(t):
        # Returns derivative of beta(t) with respect to t
        pass


class ConstantInvTempSchedule():
    def __init__(self, inverse_temp):
        self.inverse_temp = inverse_temp

    def beta(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return torch.ones_like(t) * self.inverse_temp
    
    def dbeta_dt(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return torch.zeros_like(t)