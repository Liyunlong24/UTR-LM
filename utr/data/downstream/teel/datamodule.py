# -*- coding: utf-8 -*-
# @Time    : 2024/7/15 10:55
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from typing import Union, Optional
from pathlib import Path
import os
from utr.data.alphabet import Alphabet
from utr.data.downstream.teel.dataset import TeelDataset
from utr.utils.download import download_ribosome_loading_data

class TeelDataModule(pl.LightningDataModule):
    def __init__(
        self,
        #task_type: str = 'TE',
        data_root: Union[Path, str],
        task_type: str = 'TE',
        alphabet: Alphabet = Alphabet(),
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        skip_data_preparation: bool = True,
    ):
        super().__init__()
        self.task_type = task_type
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
        dataset = TeelDataset(self.data_root, alphabet=self.alphabet, self.task_type)

        self.train_dataset, self.val_dataset, self.test_dataset = dataset.train_eval_split()

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
