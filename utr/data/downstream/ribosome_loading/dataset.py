import torch
from torch.utils.data import Dataset, Subset

import pandas as pd

from typing import Union
from pathlib import Path

from utr.data.alphabet import Alphabet

class RibosomeLoadingDataset(Dataset):
    def __init__(
        self,
        mrl_csv: Union[str, Path],
        alphabet: Alphabet,
        pad_to_max_len: bool = True,
    ):
        super().__init__()

        self.df = pd.read_csv(mrl_csv)
        self.df.dropna(subset=['rl'], inplace=True) # Remove entries with missing ribosome loading value

        self.alphabet = alphabet

        self.max_enc_seq_len = -1
        if pad_to_max_len:
            self.max_enc_seq_len = self.df['utr'].str.len().max() + 2

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        df_row = self.df.iloc[idx]

        seq = df_row['utr']
        seq_encoded = torch.tensor(self.alphabet.encode(seq, pad_to_len=self.max_enc_seq_len), dtype=torch.long)

        rl = torch.tensor(df_row['rl'], dtype=torch.float32)

        return seq_encoded, rl
