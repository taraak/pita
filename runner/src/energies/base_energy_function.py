import torch
from abc import ABC, abstractmethod
from pytorch_lightning.loggers import WandbLogger

from src.models.components.replay_buffer import ReplayBuffer


class BaseEnergyFunction(ABC):
    def __init__(self, dimensionality: int):
        self._dimensionality = dimensionality
        self._test_set = self.setup_test_set()

    def setup_test_set(self):
        return None

    def sample_test_set(self, num_points: int):
        if self.test_set is None:
            return None

        idxs = torch.randperm(len(self.test_set))[:num_points]
        return self.test_set[idxs]

    @property
    def dimensionality(self) -> int:
        return self._dimensionality

    @property
    def test_set(self):
        return self._test_set

    @abstractmethod
    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def log_on_epoch_end(
        self,
        latest_samples: torch.Tensor,
        latest_energies: torch.Tensor,
        replay_buffer: ReplayBuffer,
        wandb_logger: WandbLogger
    ) -> None:
        pass
