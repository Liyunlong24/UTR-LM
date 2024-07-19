import ml_collections
import numpy as np
import math
from collections import defaultdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

import logging
logger = logging.getLogger(__name__)


def masked_energy_loss(logits, target):
    energy_mask = target != 1e-9
    energy_loss = F.mse_loss(logits.squeeze(-1), target, reduction="none")
    energy_loss = (energy_loss * energy_mask).sum() / (energy_mask.sum() + 1e-5)
    return energy_loss



def UTRLMLoss(out, batch):

    # 只定义一个损失函数
    loss =  F.cross_entropy(out["logits"].transpose(1, 2), batch["label_tokens"], ignore_index=-1)

    if torch.isnan(loss).any() or torch.isinf(loss).any():
        logging.warning("mlm_loss is NaN or Inf. Skipping...")
        loss = torch.tensor(0.0, requires_grad=True)
    return loss
