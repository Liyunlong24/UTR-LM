import torch
from typing import Sequence
from torch.utils.data import Dataset, DataLoader
from utr.utils import parsers

class PretrainedDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            fasta_path,
    ):
        self.tokenizer = tokenizer

        with open(fasta_path) as f:
            fasta_str = f.read()
        input_seqs, input_descs = parsers.parse_fasta(fasta_str)

        self.docs = list(zip(input_seqs, input_descs))

    def mask_tokens(
            self,
            inputs: torch.Tensor,
            mlm_probability=0.15,
    ):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, mlm_probability)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(labels.tolist())

        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask,
                                                     dtype=torch.bool),
                                        value=0.0)
        padding_mask = (labels == self.tokenizer.padding_idx)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -1  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(
            labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_idx

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(
            labels.shape, 0.5)).bool() & masked_indices & (~indices_replaced)
        random_words = torch.randint(len(self.tokenizer),
                                     labels.shape,
                                     dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        return inputs, labels, masked_indices

    def _tokenize_input_sentence(
            self,
            input: str,
    ):
        inputs = self.tokenizer.encode(input)
        inputs = torch.LongTensor(inputs)
        return inputs

    def __len__(self):
        """ Returns the number of documents. """
        return len(self.docs)

    def __getitem__(self, idx):
        input_seq, input_desc = self.docs[idx]
        inputs = self._tokenize_input_sentence(input_seq)  # (L, )

        inputs, labels, masked_indices = self.mask_tokens(inputs)  # (L, ), (L, )


        return inputs, labels, masked_indices, input_desc

class EnsembleDataset(torch.utils.data.Dataset):
    """
        Implements ensemble of multiple datasets.
        Deterministic and stochastic filters can be applied to each dataset here.
    """

    def __init__(self,
                 datasets: Sequence[Dataset],
                 probabilities: Sequence[float],
                 epoch_len: int,
                 generator: torch.Generator = None,
                 _roll_at_init: bool = True,
                ):
        self.datasets = datasets
        self.probabilities = probabilities
        self.epoch_len = epoch_len
        self.generator = generator

        self._samples = [self.looped_samples(i) for i in range(len(self.datasets))]
        if _roll_at_init:
            self.reroll()

    def looped_shuffled_dataset_idx(self, dataset_len):
        while True:
            # Uniformly shuffle each dataset's indices
            weights = [1. for _ in range(dataset_len)]
            shuf = torch.multinomial(
                torch.tensor(weights),
                num_samples=dataset_len,
                replacement=False,
                generator=self.generator,
            )
            for idx in shuf:
                yield idx

    def looped_samples(self, dataset_idx):
        max_cache_len = int(self.epoch_len * self.probabilities[dataset_idx])
        dataset = self.datasets[dataset_idx]
        idx_iter = self.looped_shuffled_dataset_idx(len(dataset))
        while True:
            weights = []
            idx = []
            for _ in range(max_cache_len):
                candidate_idx = next(idx_iter)
                p = 1.0 # always sample
                weights.append([1. - p, p])
                idx.append(candidate_idx)

            samples = torch.multinomial(
                torch.tensor(weights),
                num_samples=1,
                generator=self.generator,
            )
            samples = samples.squeeze()

            cache = [i for i, s in zip(idx, samples) if s]

            for datapoint_idx in cache:
                yield datapoint_idx

    def __getitem__(self, idx):
        dataset_idx, datapoint_idx = self.datapoints[idx]
        return self.datasets[dataset_idx][datapoint_idx]

    def __len__(self):
        return self.epoch_len

    def reroll(self):
        dataset_choices = torch.multinomial(
            torch.tensor(self.probabilities),
            num_samples=self.epoch_len,
            replacement=True,
            generator=self.generator,
        )
        self.datapoints = []
        for dataset_idx in dataset_choices:
            samples = self._samples[dataset_idx]
            datapoint_idx = next(samples)
            self.datapoints.append((dataset_idx, datapoint_idx))