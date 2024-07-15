import ml_collections as mlc

import logging
import re
import copy
import yaml

from typing import Optional, Sequence, Any, Union
from utr.data.alphabet import Alphabet
from utr.data.constants import *


#model的config
def model_config(name):
    c = copy.deepcopy(default_config)

    if name == "nano":
        c.globals.embed_dim = 320

        c.model.transformer.num_blocks = 6
        c.model.transformer.num_heads = 20
    elif name == "micro":
        c.globals.embed_dim = 480

        c.model.transformer.num_blocks = 12
        c.model.transformer.num_heads = 20
    elif name == "mega":
        c.globals.embed_dim = 640

        c.model.transformer.num_blocks = 30
        c.model.transformer.num_heads = 20
    elif name == "giga":
        c.globals.embed_dim = 1280

        c.model.transformer.num_blocks = 33
        c.model.transformer.num_heads = 20

        c.training.optimizer.lr = 5e-5
        c.training.lr_scheduler.cosine_decay.eta_min = 1e-5
    else:
        raise ValueError("Invalid configuration name!")
    #确保分词器与config中的一致
    assert not any_tokenizer_discrepancies(c), "Found discrepancies in tokenizer configuration!"

    return c

def any_tokenizer_discrepancies(config):
    alphabet = Alphabet(**config['alphabet'])

    if alphabet.get_idx(MASK_TKN) != config['globals'].mask_tkn_idx:
        return True
    
    if alphabet.get_idx(PAD_TKN) != config['globals'].pad_tkn_idx:
        return True
    
    if len(alphabet) != config['globals'].alphabet_size:
        return True
    
    return False

embed_dim = mlc.FieldReference(480, field_type=int)

default_alphabet = Alphabet()
alphabet_size = mlc.FieldReference(len(default_alphabet), field_type=int)
mask_tkn_idx = mlc.FieldReference(default_alphabet.get_idx(MASK_TKN), field_type=int)
pad_tkn_idx = mlc.FieldReference(default_alphabet.get_idx(PAD_TKN), field_type=int)

mask_ratio = mlc.FieldReference(0.15, field_type=float)
mask_tkn_prob = mlc.FieldReference(0.8, field_type=float)

default_config = mlc.ConfigDict(
    {
        "globals": {
            "embed_dim": embed_dim,
            "alphabet_size": alphabet_size,
            "mask_tkn_idx": mask_tkn_idx,
            "pad_tkn_idx": pad_tkn_idx,
            "mask_ratio": mask_ratio,
            "mask_tkn_prob": mask_tkn_prob,
        },
        "alphabet": {
            "standard_tkns": RNA_TOKENS,
            "special_tkns": [CLS_TKN, PAD_TKN, EOS_TKN, UNK_TKN, MASK_TKN],
        },
        "training": {
            "optimizer": {
                "lr": 1e-4,
                "weight_decay": 0.01,
            },
            "lr_scheduler": {
                "warm_up": {
                    "iters": 2000,
                },
                "cosine_decay": {
                    "T_max": 200_000,
                    "eta_min": 1e-5,
                },
            },
            "masking": {
                "bert_masking": {
                    "mask_ratio": mask_ratio,
                    "mask_tkn_prob": mask_tkn_prob,
                    "random_tkn_prob": 0.1,
                }
            },
        },
        "model": {
            "embedding": {
                "num_embeddings": alphabet_size,
                "embedding_dim": embed_dim,
                "padding_idx": pad_tkn_idx,
            },
            "token_dropout": {
                "active": True,
                "mask_ratio": mask_ratio,
                "mask_tkn_prob": mask_tkn_prob,
                "mask_tkn_idx": mask_tkn_idx,
                "pad_tkn_idx": pad_tkn_idx,
            },
            "transformer": {
                "embed_dim": embed_dim,
                "num_blocks": 12,
                "num_heads": 20,
                "use_rot_emb": True,
                "attn_qkv_bias": False,
                "attention_dropout": 0.1,
                "transition_dropout": 0.0,
                "residual_dropout": 0.1,
                "transition_factor": 4,
                "use_flash_attn": True,
            },
            "lm_mask_head": {
                "embed_dim": embed_dim,
                "alphabet_size": alphabet_size,
            }
        }
    }
)

#以下为从utr移植过来的
def set_inf(c, inf):
    for k, v in c.items():
        if isinstance(v, mlc.ConfigDict):
            set_inf(v, inf)
        elif k == "inf":
            c[k] = inf


def recursive_set(target, src):
    """
        Recursively set target dict using src dict by matching keys and depths
    """
    for (k, v) in src.items():
        if not isinstance(v, dict):
            target[k] = v
        else:
            if k not in target:
                logging.warning(
                    f"Key '{k}' not found in target dict, but the value is a dict. "
                    f"Creating a new dict in target."
                )
                target[k] = v
            else:
                recursive_set(target[k], v)


def UTRLMConfig(
    name="finetuning",
    train=False,
    low_prec=False,
    yaml_config=None,
):
    c = copy.deepcopy(config)

    if name == "pretraining":
        pass

    if name == "finetuning":
        pass

    if train:
        pass

    if low_prec:
        pass

    if yaml_config is not None:
        logging.info(f"Loading yaml config from {yaml_config}")
        with open(yaml_config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        if yaml_config is not None:
            recursive_set(c, yaml_config)
        else:
            logging.warning("The yaml config is empty!")
    return c


STRING_PLACEHOLDER = "string_placeholder"

# 定义了loss、optimizer、scheduler、ema
config = mlc.ConfigDict({
    "globals": {
    },
    #data和model不确定用了没有
    "data": {
        "max_len": 512,
        "batch_size": 2,
        "num_workers": 16,
    },
    "model": {
        "num_layers": 6,
        "embed_dim": 256,
        "attention_heads": 16,
        "token_dropout": True,
        "has_energy_head": False,
        "has_ss_head": False,
    },
    "loss": {
        "mlm_loss": {
            "weight": 1.0,
        },
    },
    "optimizer": {
        "lr": 0.001,
        "weight_decay": 0.00000,
        "eps": 1e-5,
        "lr_ratio": 1.0,
    },
    "scheduler": {
        "warmup_no_steps": 1000,
        "start_decay_after_n_steps": 50000,
        "decay_every_n_steps": 50000,
        "decay_factor": 0.95,
        "type": "alphafold"
    },
    "ema": {"decay": 0.999},
})