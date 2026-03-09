# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 16:16:18 2026

@author: kubota
"""

# engine/builder.py
from __future__ import annotations

import os
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from model.condnet_tart import UNet_2D, CondNet_Transfer


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
        if not isinstance(model, nn.DataParallel):
            _log(logger, "Enabling nn.DataParallel")
            model = nn.DataParallel(model)
    return model


def _strip_module_prefix_if_needed(state_dict: dict) -> dict:
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _load_unet_full_weights(
    unet_model: nn.Module,
    ckpt_path: str,
    device: torch.device,
    logger=None,
    strict: bool = True,
):
    """
    segmentation 学習済み UNet_2D の state_dict をそのまま読み込む想定。
    （昔のコード: state_dict = torch.load(...); model.load_state_dict(state_dict)）
    """
    if not ckpt_path:
        return
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"pretrained_unet_path not found: {ckpt_path}")

    _log(logger, f"Loading pretrained UNet_2D from: {ckpt_path}")

    sd = torch.load(ckpt_path, map_location=device, weights_only=True)

    # もし {"model": ...} 形式なら吸収（ただし segmentation pth は普通これじゃない）
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]

    sd = _strip_module_prefix_if_needed(sd)

    core = unet_model.module if isinstance(unet_model, nn.DataParallel) else unet_model
    core.load_state_dict(sd, strict=strict)

    _log(logger, f"UNet_2D weights loaded (strict={strict}).")


def _apply_freeze_unet_by_keywords(
    unet_model: nn.Module,
    freeze_keywords: list[str],
    logger=None,
):
    """
    元コード互換: name にキーワードが含まれていたら requires_grad=False
    """
    if not freeze_keywords:
        return

    core = unet_model.module if isinstance(unet_model, nn.DataParallel) else unet_model

    n_total = 0
    n_frozen = 0
    for name, param in core.named_parameters():
        n_total += param.numel()
        if any(k in name for k in freeze_keywords):
            param.requires_grad = False
            n_frozen += param.numel()
        else:
            param.requires_grad = True

    _log(logger, f"Freeze applied to UNet_2D by keywords={freeze_keywords}")
    _log(logger, f"Frozen params: {n_frozen:,} / {n_total:,}")


def _build_optimizer(cfg: Dict[str, Any], model, logger=None):
    """
    dictモデル: {"original":..., "ctnet":...} -> optimizer dict
    単一モデル: nn.Module -> optimizer
    """
    train_cfg = cfg.get("train", {})
    wd = float(train_cfg.get("weight_decay", 0.0))
    opt_name = str(train_cfg.get("optimizer", "adam")).lower()

    def _trainable_params(m: nn.Module):
        return [p for p in m.parameters() if p.requires_grad]

    def _make_opt(params, lr):
        if opt_name == "adam":
            return optim.Adam(params, lr=lr, weight_decay=wd)
        if opt_name == "adamw":
            return optim.AdamW(params, lr=lr, weight_decay=wd)
        if opt_name == "sgd":
            momentum = float(train_cfg.get("momentum", 0.9))
            return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
        raise ValueError(f"Unknown optimizer: {opt_name}")

    if isinstance(model, dict):
        if ("original" not in model) or ("ctnet" not in model):
            raise KeyError("When model is dict, keys must include 'original' and 'ctnet'.")

        lr_o = float(train_cfg.get("lr_original", train_cfg.get("lr", 1e-4)))
        lr_c = float(train_cfg.get("lr_ctnet", train_cfg.get("lr", 1e-3)))

        m_o = model["original"]
        m_c = model["ctnet"]

        params_o = _trainable_params(m_o)
        params_c = _trainable_params(m_c)

        n_all_o = sum(p.numel() for p in m_o.parameters())
        n_tr_o = sum(p.numel() for p in params_o)
        n_all_c = sum(p.numel() for p in m_c.parameters())
        n_tr_c = sum(p.numel() for p in params_c)

        _log(logger, f"[original] Trainable params: {n_tr_o:,} / {n_all_o:,} | lr={lr_o:g}")
        _log(logger, f"[ctnet]    Trainable params: {n_tr_c:,} / {n_all_c:,} | lr={lr_c:g}")

        return {
            "original": _make_opt(params_o, lr_o),
            "ctnet": _make_opt(params_c, lr_c),
        }

    lr = float(train_cfg.get("lr", 1e-3))
    params = _trainable_params(model)

    n_all = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in params)
    _log(logger, f"Trainable params: {n_tr:,} / {n_all:,} | lr={lr:g}")

    return _make_opt(params, lr)


def _detach_unet_outputs(out_old: Any) -> Any:
    """
    UNet_2D出力(dict)のdetach対応。
    """
    if isinstance(out_old, dict):
        out2 = {}
        for k, v in out_old.items():
            if torch.is_tensor(v):
                out2[k] = v.detach()
            else:
                out2[k] = v
        return out2
    if torch.is_tensor(out_old):
        return out_old.detach()
    return out_old


class ChainModel(nn.Module):
    """
    multi_model=False のとき用:
      forward(img1,img2,mask) -> original(...) -> ctnet(out_old) -> out
    """
    def __init__(self, original: nn.Module, ctnet: nn.Module, train_original: bool = True):
        super().__init__()
        self.original = original
        self.ctnet = ctnet
        self.train_original = train_original

    def forward(self, img1: torch.Tensor, img2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out_old = self.original(img1, img2, mask)
        if not self.train_original:
            out_old = _detach_unet_outputs(out_old)
        out = self.ctnet(out_old)
        return out


def build(cfg: Dict[str, Any], logger=None):
    """
    Returns:
      - multi_model=True : (models_dict, device, optimizers_dict)
      - multi_model=False: (single_model, device, optimizer)
    """
    device = _select_device(cfg, logger=logger)

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})

    multi = bool(model_cfg.get("multi_model", False))

    # ---- UNet init args from cfg ----
    num_classes = int(model_cfg.get("num_classes", 10))
    transformer_count = int(model_cfg.get("transformer_count", 6))

    # ---- freeze keywords ----
    freeze_keywords = model_cfg.get("freeze_keywords", None)
    if freeze_keywords is None:
        freeze_keywords = ["TCB1", "TCB2", "TCB3", "TCB4", "TCB5", "Trans"]
    if not isinstance(freeze_keywords, list):
        raise ValueError("model.freeze_keywords must be a list of strings (or omitted).")

    # ---- pretrained segmentation UNet path ----
    pretrained = str(model_cfg.get("pretrained_unet_path", "")).strip()

    # ---- whether to train original ----
    train_original = bool(train_cfg.get("train_original", True))

    # ==========================================================
    # multi_model=True: original/ctnet を別々に持って trainer側で2opt運用
    # ==========================================================
    if multi:
        original = UNet_2D(num_classes=num_classes, transformer_count=transformer_count).to(device)
        ctnet = CondNet_Transfer().to(device)

        # DPはこのモードでは individual に掛けてOK
        original = _maybe_dataparallel(original, device, cfg, logger=logger)
        ctnet = _maybe_dataparallel(ctnet, device, cfg, logger=logger)

        if pretrained:
            _load_unet_full_weights(original, pretrained, device, logger=logger, strict=True)

        _apply_freeze_unet_by_keywords(original, freeze_keywords, logger=logger)

        models = {"original": original, "ctnet": ctnet}
        optimizer = _build_optimizer(cfg, models, logger=logger)
        return models, device, optimizer

    # ==========================================================
    # multi_model=False: chain 1本化（DPは chain に1回だけ）
    # ==========================================================
    original = UNet_2D(num_classes=num_classes, transformer_count=transformer_count).to(device)
    ctnet = CondNet_Transfer().to(device)

    if pretrained:
        _load_unet_full_weights(original, pretrained, device, logger=logger, strict=True)

    _apply_freeze_unet_by_keywords(original, freeze_keywords, logger=logger)

    chain = ChainModel(original=original, ctnet=ctnet, train_original=train_original).to(device)
    chain = _maybe_dataparallel(chain, device, cfg, logger=logger)

    optimizer = _build_optimizer(cfg, chain, logger=logger)
    return chain, device, optimizer