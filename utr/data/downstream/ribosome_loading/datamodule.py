from torch.utils.data import DataLoader

import pytorch_lightning as pl

from typing import Union, Optional
from pathlib import Path
import os
from utr.data.alphabet import Alphabet
from utr.data.downstream.ribosome_loading.dataset import RibosomeLoadingDataset
from utr.utils.download import download_ribosome_loading_data

TRAIN_PATH = "4.1_train.csv"
VAL_PATH = "4.1_val_test.csv"
TEST_PATH = "4.1_val_test.csv"

class RibosomeLoadingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: Union[Path, str],
        alphabet: Alphabet = Alphabet(),
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        skip_data_preparation: bool = True,
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.alphabet = alphabet

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.skip_data_preparation = skip_data_preparation
        self._data_prepared = skip_data_preparation

    def prepare_data(self):
        if not self.skip_data_preparation and not self._data_prepared:
            download_ribosome_loading_data(self.data_root)
            self._data_prepared = True

    def setup(self, stage: Optional[str] = None):

        self.train_dataset = RibosomeLoadingDataset(self.data_root / TRAIN_PATH, alphabet=self.alphabet)
        self.val_dataset = RibosomeLoadingDataset(self.data_root / VAL_PATH, alphabet=self.alphabet)
        self.test_dataset = RibosomeLoadingDataset(self.data_root / TEST_PATH, alphabet=self.alphabet)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
        )
