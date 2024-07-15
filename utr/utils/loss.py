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


class UTRLMLoss(nn.Module):
    def __init__(self, config):
        super(UTRLMLoss, self).__init__()
        self.config = config

    def loss(self, out, batch, _return_breakdown=False):
        loss_fns = {}
        if "mlm_loss" in self.config:#成立
            loss_fns["mlm_loss"] = lambda: F.cross_entropy(out["logits"].transpose(1, 2), batch["label_tokens"], ignore_index=-1)
       # if "energy_loss" in self.config:
        #    loss_fns["energy_loss"] = lambda: masked_energy_loss(out["logits_energy"], batch["label_energy"])
       # if "secstr_loss" in self.config:
          #  loss_fns["secstr_loss"] = lambda: F.cross_entropy(out["logits_ss"].transpose(1, 2), batch["label_secstr"], ignore_index=-1)

        cum_loss = 0.
        losses = {}
        for loss_name, loss_fn in loss_fns.items():
            if loss_name not in self.config:
                continue
            weight = self.config[loss_name].weight
            if weight == 0:
                continue
            loss = loss_fn()

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logging.warning(f"{loss_name} loss is NaN. Skipping...")
                loss = loss.new_tensor(0., requires_grad=True)
            cum_loss = cum_loss + weight * loss
            losses[loss_name] = loss.detach().clone()

        losses["loss"] = cum_loss.detach().clone()

        if not _return_breakdown:
            return cum_loss

        return cum_loss, losses
    
    def forward(self, out, batch, _return_breakdown=False):
        if not _return_breakdown:
            cum_loss = self.loss(out, batch, _return_breakdown)
            return cum_loss
        else:
            cum_loss, losses = self.loss(out, batch, _return_breakdown)
            return cum_loss, losses