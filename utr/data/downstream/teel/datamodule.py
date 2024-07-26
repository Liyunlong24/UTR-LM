from torch.utils.data import DataLoader

import pytorch_lightning as pl

from typing import Union, Optional
from pathlib import Path
import os
from utr.data.alphabet import Alphabet
from utr.data.downstream.teel.dataset import TeelDataset

class TeelDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: Union[Path, str],
        alphabet: Alphabet = Alphabet(),
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        task_type: str = "TE",
    ):
        super().__init__()

        self.dataset = Path(data_root)
        self.alphabet = alphabet

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.task_type = task_type


    def setup(self, stage: Optional[str] = None):
        print(f"The task_tpye is {self.task_type},The dataset processed is {self.dataset}")
        self.train_dataset = TeelDataset(Path(f"./data/{self.dataset}_train_data.csv"), alphabet=self.alphabet, task_type=self.task_type,)
        self.val_dataset = TeelDataset(Path(f"./data/{self.dataset}_val_data.csv"), alphabet=self.alphabet, task_type=self.task_type,)
        self.test_dataset = TeelDataset(Path(f"./data/{self.dataset}_test_data.csv"), alphabet=self.alphabet, task_type=self.task_type,)

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

