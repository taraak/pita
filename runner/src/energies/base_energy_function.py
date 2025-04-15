from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from pytorch_lightning.loggers import WandbLogger
from src.models.components.replay_buffer import ReplayBuffer
from src.utils.data_utils import remove_mean


class BaseEnergyFunction(ABC):
    def __init__(
        self,
        dimensionality: int,
        n_particles: Optional[int] = None,
        spatial_dim: Optional[int] = None,
        is_molecule: Optional[bool] = False,
        normalization_min: Optional[float] = None,
        normalization_max: Optional[float] = None,
    ):
        self._dimensionality = dimensionality
        self._is_molecule = is_molecule

        self._test_set = self.setup_test_set()
        self._val_set = self.setup_val_set()
        self._train_set = None

        self.normalization_min = normalization_min
        self.normalization_max = normalization_max

    def setup_test_set(self) -> Optional[torch.Tensor]:
        return None

    def setup_train_set(self) -> Optional[torch.Tensor]:
        return None

    def setup_val_set(self) -> Optional[torch.Tensor]:
        return None
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_molecule:
            return self._normalize_molecule(x)
        return self._normalize(x)
    
    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_molecule:
            return self._unnormalize_molecule(x)
        return self._unnormalize(x)
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        assert self.normalization_min is not None and self.normalization_max is not None, "Normalization min and max should be set"
        mins = self.normalization_min
        maxs = self.normalization_max
        ## [ 0, 1 ]
        x = (x - mins) / (maxs - mins + 1e-5)
        ## [ -1, 1 ]
        return x * 2 - 1

    def _unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        assert self.normalization_min is not None and self.normalization_max is not None, "Normalization min and max should be set"
        mins = self.normalization_min
        maxs = self.normalization_max
        x = (x + 1) / 2
        return x * (maxs - mins) + mins

    def _normalize_molecule(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self._dimensionality
        assert self.data_normalization_factor is not None, "Standard deviation should be computed first"
        x = remove_mean(x, self.n_particles, self.n_spatial_dim)
        x = x / self.data_normalization_factor
        return x

    def _unnormalize_molecule(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self._dimensionality
        assert self.data_normalization_factor is not None, "Standard deviation should be computed first"
        x = x * self.data_normalization_factor
        return x

    def sample_test_set(self, num_points: int, normalize: bool = False) -> Optional[torch.Tensor]:
        if self.test_set is None:
            return None

        outs = self.test_set[:num_points]
        if normalize:
            outs = self.normalize(outs)
        return outs

    def sample_train_set(self, num_points: int, normalize: bool = False) -> Optional[torch.Tensor]:
        if self.train_set is None:
            self._train_set = self.setup_train_set()
        outs = self.train_set[:num_points]
        if normalize:
            outs = self.normalize(outs)

        return outs

    def sample_val_set(self, num_points: int, normalize: bool = False) -> Optional[torch.Tensor]:
        if self.val_set is None:
            return None

        outs = self.val_set[:num_points]
        if normalize:
            outs = self.normalize(outs)

        return outs

    @property
    def dimensionality(self) -> int:
        return self._dimensionality

    @property
    def is_molecule(self) -> Optional[bool]:
        return self._is_molecule

    @property
    def test_set(self) -> Optional[torch.Tensor]:
        return self._test_set

    @property
    def val_set(self) -> Optional[torch.Tensor]:
        return self._val_set

    @property
    def train_set(self) -> Optional[torch.Tensor]:
        return self._train_set

    @abstractmethod
    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def score(self, samples: torch.Tensor) -> torch.Tensor:
        grad_fxn = torch.func.grad(self.__call__)
        vmapped_grad = torch.vmap(grad_fxn)
        return vmapped_grad(samples)

    def log_on_epoch_end(
        self,
        latest_samples: torch.Tensor,
        latest_energies: torch.Tensor,
        replay_buffer: ReplayBuffer,
        wandb_logger: WandbLogger,
    ) -> None:
        pass

    def save_samples(
        self,
        samples: torch.Tensor,
        dataset_name: str,
    ) -> None:
        np.save(f"{dataset_name}_samples.npy", samples.cpu().numpy())
