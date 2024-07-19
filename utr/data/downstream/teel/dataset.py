import torch
from torch.utils.data import Dataset, Subset

import pandas as pd

from typing import Union
from pathlib import Path

from utr.data.alphabet import Alphabet
from sklearn.model_selection import train_test_split


MIN_EVAL_SEQ_LEN = 25
MAX_EVAL_SEQ_LEN = 100


def _generate_eval_set_eq_sampled_lens(df: pd.DataFrame, min_seq_len: int, max_seq_len: int, num_samples_per_len: int):
    eval_idcs = set()

    for len in range(min_seq_len, max_seq_len + 1):
        len_df = df[df['len'] == len].copy()
        len_df.sort_values('total_reads', inplace=True, ascending=False)
        eval_idcs.update(len_df.iloc[:num_samples_per_len].index.values.tolist())

    return eval_idcs


class TeelDataset(Dataset):
    def __init__(
            self,
            mrl_csv: Union[str, Path],
            alphabet: Alphabet,
            task_type:str = 'TE',
            pad_to_max_len: bool = True,
    ):
        super().__init__()

        self.df = pd.read_csv(mrl_csv)
        self.df.dropna(subset=['te_log'], inplace=True)  # Remove entries with missing ribosome loading value
        self.df.dropna(subset=['rnaseq_log'], inplace=True)
        self.alphabet = alphabet
        self.task_type = task_type
        self.max_enc_seq_len = -1
        if pad_to_max_len:
            self.max_enc_seq_len = self.df['utr'].str.len().max() + 2

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        assert self.task_type =='TE' or self.task_type == 'EL' , 'task_type must be TE or EL'
        df_row = self.df.iloc[idx]

        seq = df_row['utr']
        seq_encoded = torch.tensor(self.alphabet.encode(seq, pad_to_len=self.max_enc_seq_len), dtype=torch.long)
        if self.task_type == 'TE':
            rl = torch.tensor(df_row['te_log'], dtype=torch.float32)
        elif self.task_type == 'EL':
            rl = torch.tensor(df_row['rnaseq_log'], dtype=torch.float32)

        return seq_encoded, rl
    def train_eval_split(self, val_size: float = 0.1, test_size: float = 0.1):
        assert 'te_log' in self.df.columns and 'rnaseq_log' in self.df.columns, "CSV文件缺少te_log列或者rnaseq_log列"

        self.df.drop_duplicates('utr', inplace=True, keep=False)
        self.df.reset_index(inplace=True, drop=True)

        # 获取数据集的所有索引
        all_indices = list(range(len(self.df)))

        # 首先按照7:3比例切分为训练集+验证集 和 测试集
        train_val_indices, test_indices = train_test_split(all_indices, test_size=test_size, random_state=42)

        # 在训练集+验证集内部按照训练集和验证集比例切分
        train_indices, val_indices = train_test_split(train_val_indices, test_size=val_size / (1 - test_size),
                                                      random_state=42)

        train_ds = Subset(self, indices=train_indices)
        val_ds = Subset(self, indices=val_indices)
        test_ds = Subset(self, indices=test_indices)

        return train_ds, val_ds, test_ds


