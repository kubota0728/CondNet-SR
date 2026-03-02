# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 11:40:19 2026

@author: kubota
"""

import torch
import torch.nn as nn


class MAELoss(nn.Module):
    def forward(self, pred, target, batch=None):
        return torch.mean(torch.abs(pred - target))


class MixedLogMAELoss(nn.Module):
    """
    0.95*MAE + 0.05*logMAE
    """
    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, pred, target, batch=None):
        err = torch.abs(pred - target)
        mae = torch.mean(err)
        logmae = torch.mean(torch.log(err + self.eps))
        return 0.95 * mae + 0.05 * logmae


class CondNetCSLoss(nn.Module):
    def forward(self, pred, target, batch=None):
        # 仮：mask付きMAE（必要なら拡張）
        if batch is not None and "mask" in batch:
            m = batch["mask"]
            err = torch.abs(pred - target) * m
            denom = torch.clamp(m.sum(), min=1.0)
            return err.sum() / denom
        return torch.mean(torch.abs(pred - target))


def build_loss(cfg):
    name = cfg.get("loss", {}).get("name", "mae").lower()

    if name == "mae":
        return MAELoss()
    if name == "logmae":
        return MixedLogMAELoss()
    if name == "condnet_cs":
        return CondNetCSLoss()

    raise ValueError(f"Unknown loss: {name}")