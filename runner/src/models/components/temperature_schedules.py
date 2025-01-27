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


class LinearInvTempSchedule(BaseInverseTempSchedule):
    def __init__(self, inverse_temp, inverse_temp_start=1.0):
        self.inverse_temp = inverse_temp
        self.inverse_temp_start = inverse_temp_start

    def beta(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return self.inverse_temp + (self.inverse_temp_start - self.inverse_temp) * t
        
    
    def dbeta_dt(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return torch.ones_like(t) * (self.inverse_temp_start- self.inverse_temp)
    

class ExponentialInvTempSchedule(BaseInverseTempSchedule):
    def __init__(self, inverse_temp, inverse_temp_start=1.0):
        self.inverse_temp = inverse_temp
        self.inverse_temp_start = inverse_temp_start

    def beta(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return self.inverse_temp * torch.exp(np.log(self.inverse_temp_start/self.inverse_temp) * t)
    
    def dbeta_dt(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return np.log(self.inverse_temp_start/self.inverse_temp) * self.inverse_temp * torch.exp(np.log(self.inverse_temp_start/self.inverse_temp) * t)


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