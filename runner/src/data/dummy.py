import numpy as np
from lightning import LightningDataModule
from torch.utils.data import DataLoader


class DummyDataModule(LightningDataModule):
    def __init__(self, n_batches_per_epoch: int = 100):
        super().__init__()
        self.n_batches_per_epoch = n_batches_per_epoch

    def train_dataloader(self):
        return DataLoader(np.arange(self.n_batches_per_epoch)[:, None])

    def val_dataloader(self):
        return DataLoader(np.arange(1)[:, None])

    def test_dataloader(self):
        return DataLoader(np.arange(100)[:, None])
