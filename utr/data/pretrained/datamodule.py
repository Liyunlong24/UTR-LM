import torch
import pytorch_lightning as pl

from functools import partial
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from utr.utils.data import utrlm_BatchConverter
from utr.data.pretrained.dataset import PretrainedDataset,EnsembleDataset

# Default train/val/test directory names
TRAIN_DIR_NAME = "train"
VAL_DIR_NAME = "valid"
TEST_DIR_NAME = "test"

class PretrainedDataModule(pl.LightningDataModule):
    def __init__(
            self,
            tokenizer: object,
            max_len: int,
            batch_size: int,
            num_workers: int,
            train_epoch_len: int,
            train_path: Optional[str] = None,
            valid_path: Optional[str] = None,
            test_path: Optional[str] = None,
            batch_seed: Optional[int] = None,
            **kwargs,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.batch_seed = batch_seed
        self.train_epoch_len = train_epoch_len

        if train_path is None and valid_path is None and test_path is None:
            raise ValueError(
                "At one valid path should be given for either train, valid or test"
            )
    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        dataset_gen = partial(
            PretrainedDataset,
            tokenizer=self.tokenizer,
        )

        generator = None
        if self.batch_seed is not None:
            generator = torch.Generator()
            generator = generator.manual_seed(self.batch_seed + 1)

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        if self.train_path is not None:
            train_dataset = dataset_gen(fasta_path=self.train_path)
            self.train_dataset = EnsembleDataset(
                datasets=[train_dataset],
                probabilities=[1.],
                epoch_len=self.train_epoch_len,
                generator=generator,
                _roll_at_init=False,
            )

        if self.valid_path is not None:
            self.valid_dataset = dataset_gen(fasta_path=self.valid_path)
        if self.test_path is not None:
            self.test_dataset = dataset_gen(fasta_path=self.test_path)

    def dataloader_gen(self, stage):
        generator = None
        if self.batch_seed is not None:
            generator = torch.Generator()
            generator = generator.manual_seed(self.batch_seed)

        if stage == "train":
            dataset = self.train_dataset
            dataset.reroll()
        elif stage == "valid":
            dataset = self.valid_dataset
        elif stage == "test":
            dataset = self.test_dataset
        else:
            raise ValueError("Invalid stage")

        dl = DataLoader(
            dataset,
            generator=generator,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=utrlm_BatchConverter(self.tokenizer, self.max_len),
        )
        #print(dl)

        return dl

    def train_dataloader(self):
        if self.train_dataset is not None:
            return self.dataloader_gen("train")
        return None

    def val_dataloader(self):
        if self.valid_dataset is not None:
            return self.dataloader_gen("valid")
        return None

    def test_dataloader(self):
        if self.test_dataset is not None:
            return self.dataloader_gen("test")
        return None
