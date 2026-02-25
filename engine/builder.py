# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 16:16:18 2026

@author: kubota
"""

# engine/builder.py
from __future__ import annotations

import os
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from model.condnet_tart import CondNetTART


def _log(logger, msg: str):
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


def _select_device(cfg: Dict[str, Any], logger=None) -> torch.device:
    run_cfg = cfg.get("run", {})
    force_cpu = bool(run_cfg.get("force_cpu", False))

    if (not force_cpu) and torch.cuda.is_available():
        device = torch.device("cuda")
        _log(logger, f"Using CUDA ({torch.cuda.device_count()} GPU(s))")
    else:
        device = torch.device("cpu")
        _log(logger, "Using CPU")
    return device


def _maybe_dataparallel(model: nn.Module, device: torch.device, cfg: Dict[str, Any], logger=None) -> nn.Module:
    run_cfg = cfg.get("run", {})
    use_dp = bool(run_cfg.get("data_parallel", True))

    if device.type == "cuda" and use_dp and torch.cuda.device_count() > 1:
        _log(logger, "Enabling nn.DataParallel")
        model = nn.DataParallel(model)
    return model


def _load_backbone_weights(
    model: nn.Module,
    ckpt_path: str,
    device: torch.device,
    logger=None,
):
    """
    Loads weights into model.backbone (UNet_2D) from a .pth state_dict.
    Your old file saved UNet_2D.state_dict(), not a full checkpoint dict.
    So we try to load as state_dict directly.
    """
    if not ckpt_path:
        return
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"pretrained_unet_path not found: {ckpt_path}")

    _log(logger, f"Loading pretrained backbone from: {ckpt_path}")

    sd = torch.load(ckpt_path, map_location=device)

    # If somebody saved {"model": ...} style, handle it
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]

    # DataParallel mismatch safety: strip "module." if needed
    def strip_module_prefix(state_dict):
        if not any(k.startswith("module.") for k in state_dict.keys()):
            return state_dict
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    # We load only backbone part. If the state dict is for UNet_2D only,
    # it will match model.backbone.* keys, not the wrapped keys.
    # So we try loading into backbone directly first.
    core = model.module if isinstance(model, nn.DataParallel) else model
    backbone = core.backbone

    # Some checkpoints may be saved from UNet_2D directly (keys like "TCB1.conv1.weight"...)
    sd = strip_module_prefix(sd)

    missing, unexpected = backbone.load_state_dict(sd, strict=False)
    _log(logger, f"Backbone load done. missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0:
        _log(logger, f"  missing examples: {missing[:5]}")
    if len(unexpected) > 0:
        _log(logger, f"  unexpected examples: {unexpected[:5]}")


def _apply_freeze(cfg: Dict[str, Any], model: nn.Module, logger=None):
    """
    Freeze backbone parts as you did:
      TCB1..TCB5 and Trans
    Uses model.freeze_backbone_parts() if available.
    """
    model_cfg = cfg.get("model", {})
    freeze_cfg = model_cfg.get("freeze", None)
    if not freeze_cfg:
        return

    core = model.module if isinstance(model, nn.DataParallel) else model

    if hasattr(core, "freeze_backbone_parts"):
        core.freeze_backbone_parts(tuple(freeze_cfg))
        _log(logger, f"Freeze applied to backbone parts: {freeze_cfg}")
    else:
        # fallback: do name matching in backbone
        for name, param in core.backbone.named_parameters():
            param.requires_grad = not any(k in name for k in freeze_cfg)
        _log(logger, f"Freeze applied (fallback) to: {freeze_cfg}")


def _build_optimizer(cfg: Dict[str, Any], model: nn.Module, logger=None) -> optim.Optimizer:
    train_cfg = cfg.get("train", {})
    lr = float(train_cfg.get("lr", 1e-3))
    wd = float(train_cfg.get("weight_decay", 0.0))

    # trainable params only
    params = [p for p in model.parameters() if p.requires_grad]
    n_all = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in params)

    _log(logger, f"Trainable params: {n_tr:,} / {n_all:,}")

    opt_name = str(train_cfg.get("optimizer", "adam")).lower()
    if opt_name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=wd)
    if opt_name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=wd)
    if opt_name == "sgd":
        momentum = float(train_cfg.get("momentum", 0.9))
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)

    raise ValueError(f"Unknown optimizer: {opt_name}")


def build(
    cfg: Dict[str, Any],
    logger=None,
) -> Tuple[nn.Module, torch.device, optim.Optimizer]:
    """
    Returns:
      model, device, optimizer
    """
    device = _select_device(cfg, logger=logger)

    model_cfg = cfg.get("model", {})
    num_classes = int(model_cfg.get("num_classes", 10))
    transformer_count = int(model_cfg.get("transformer_count", 6))

    model = CondNetTART(num_classes=num_classes, transformer_count=transformer_count)
    model.to(device)

    # DataParallel (optional)
    model = _maybe_dataparallel(model, device, cfg, logger=logger)

    # load pretrained UNet weights (optional)
    pretrained = model_cfg.get("pretrained_unet_path", "")
    if pretrained:
        _load_backbone_weights(model, pretrained, device, logger=logger)

    # freeze (optional)
    _apply_freeze(cfg, model, logger=logger)

    # optimizer
    optimizer = _build_optimizer(cfg, model, logger=logger)

    return model, device, optimizer