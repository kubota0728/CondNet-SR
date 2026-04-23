# -*- coding: utf-8 -*-
"""
Runner entry points shared by the CLI (main.py) and the GUI (gui/).

This module owns the full train/eval pipeline driven by an in-memory config
dict. It does NOT read command-line arguments and does NOT build the logger --
callers construct those and hand them in. This makes the same pipeline
reachable from two places:

  - `main.py`  : parses YAML, sets up logger, calls run_train / run_eval.
  - `gui/`     : reads the same YAML via the SettingsDialog, runs these
                 functions inside a QThread worker with optional callbacks
                 for progress / stop / preview wired to the UI.

The callback parameters on run_train / run_eval are optional and default to
None, so CLI behaviour is unchanged.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets.dataloader import CondDataset
from engine.builder import build
from engine.trainer import Trainer


# ----------------------------------------------------------------------
# Low-level helpers (previously private in main.py)
# ----------------------------------------------------------------------

def prepare_output_dirs(cfg: Dict[str, Any], timestamp: str) -> str:
    """Create the per-run checkpoint directory and record it back in cfg."""
    out_cfg = cfg.get("output", {})
    ckpt_root = out_cfg.get("ckpt_root", "checkpoints")

    ckpt_dir = os.path.join(ckpt_root, timestamp)
    os.makedirs(ckpt_dir, exist_ok=True)

    cfg.setdefault("output", {})
    cfg["output"]["ckpt_dir"] = ckpt_dir

    return ckpt_dir


def make_loader(df: pd.DataFrame, cfg: Dict[str, Any], mode: str) -> DataLoader:
    """Build a DataLoader for the given mode (train / val / eval|test|predict)."""
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    val_cfg = cfg.get("val", {})

    image_size = int(data_cfg.get("image_size", 256))
    batch_size = int(data_cfg.get("batch_size", 32))
    val_batch_size = int(data_cfg.get("val_batch_size", batch_size))
    test_batch_size = int(data_cfg.get("test_batch_size", 1))

    if mode == "train":
        shuffle = bool(data_cfg.get("shuffle_train", True))
        num_workers = int(train_cfg.get("num_workers", 0))
        bs = batch_size
        augmentation = True
    elif mode == "val":
        shuffle = bool(data_cfg.get("shuffle_val", False))
        num_workers = int(val_cfg.get("num_workers", 0))
        bs = val_batch_size
        augmentation = False
    elif mode in ("eval", "test", "predict"):
        shuffle = False
        num_workers = int(val_cfg.get("num_workers", 0))
        bs = test_batch_size
        augmentation = False
    else:
        raise ValueError(f"Unknown loader mode: {mode}")

    ds = CondDataset(df, image_size=image_size, augmentation=augmentation)
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


def resolve_ckpt_path_for_mode(cfg: Dict[str, Any], mode: str) -> str:
    """
    Eval-like modes prefer `eval.load_pth`; fall back to `run.load_pth`.
    Train mode uses `run.load_pth` for resuming.
    """
    run_cfg = cfg.get("run", {})
    eval_cfg = cfg.get("eval", {})

    if mode in ("eval", "test", "predict"):
        p = str(eval_cfg.get("load_pth", "")).strip()
        if p:
            return p
        return str(run_cfg.get("load_pth", "")).strip()

    return str(run_cfg.get("load_pth", "")).strip()


def load_checkpoint_into_model(
    model,
    ckpt_path: str,
    device: torch.device,
    logger,
    strict: bool = True,
) -> None:
    """Load weights into a single model or a dict-of-models (original + ctnet)."""
    if not ckpt_path:
        raise ValueError("Checkpoint path is empty.")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Specified pth not found: {ckpt_path}")

    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt

    def _load_one(m, sd):
        if isinstance(m, torch.nn.DataParallel):
            m.module.load_state_dict(sd, strict=strict)
        else:
            m.load_state_dict(sd, strict=strict)

    if isinstance(model, dict):
        if not (isinstance(state, dict) and "original" in state and "ctnet" in state):
            raise ValueError("Checkpoint format mismatch: expected model={'original':..., 'ctnet':...}")
        _load_one(model["original"], state["original"])
        _load_one(model["ctnet"], state["ctnet"])
    else:
        _load_one(model, state)

    logger.info("Model weights loaded successfully.")


# ----------------------------------------------------------------------
# Top-level entry points
# ----------------------------------------------------------------------

def run_train(
    cfg: Dict[str, Any],
    logger,
    ckpt_dir: str,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    preview_cb: Optional[Callable[[int, list], None]] = None,
    preview_cases: Optional[list] = None,
    batch_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    """
    Run a full training pass.

    Args:
        cfg: parsed YAML config (dict)
        logger: already-configured logger
        ckpt_dir: target directory for checkpoints (typically from
                  prepare_output_dirs)
        progress_cb: optional callable invoked at each epoch end with a dict
                     containing epoch / train_loss / val_loss / elapsed_seconds
                     / best_loss / is_best. Used by the GUI for live progress,
                     ETA and loss curve. CLI passes None.
        stop_check: optional zero-arg callable; if it returns True the trainer
                    finishes the current epoch and then stops cleanly.
        preview_cb: optional callable(epoch, samples) where each sample is a
                    dict of t1/t2/pred/gt numpy arrays plus pid/slice labels.
                    Called once per epoch for GUI sample-prediction preview.
        preview_cases: list of (ixi_id: str, slice: int) tuples. Only these
                       val samples are captured for preview_cb.

    Behavior when all callbacks are None is identical to the prior CLI flow.
    """
    data_cfg = cfg.get("data", {})
    train_csv = data_cfg.get("train_csv", None)
    val_csv = data_cfg.get("val_csv", None)

    if not train_csv or not val_csv:
        raise ValueError("For train mode, train_csv and val_csv are required.")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    logger.info(f"Train df: {len(train_df)} rows")
    logger.info(f"Val df: {len(val_df)} rows")

    model, device, optimizer = build(cfg, logger=logger)

    train_loader = make_loader(train_df, cfg, mode="train")
    val_loader = make_loader(val_df, cfg, mode="val")

    # Resume if run.load_pth is set
    ckpt_path = resolve_ckpt_path_for_mode(cfg, mode="train")
    if ckpt_path:
        load_checkpoint_into_model(model, ckpt_path, device, logger, strict=True)

    trainer = Trainer(
        cfg=cfg,
        model=model,
        device=device,
        optimizer=optimizer,
        logger=logger,
        ckpt_dir=ckpt_dir,
    )
    trainer.fit(
        train_loader,
        val_loader,
        progress_cb=progress_cb,
        stop_check=stop_check,
        preview_cb=preview_cb,
        preview_cases=preview_cases,
        batch_cb=batch_cb,
    )


def run_eval(
    cfg: Dict[str, Any],
    logger,
    ckpt_dir: str,
    progress_cb: Optional[Callable[..., None]] = None,
) -> None:
    """
    Run an eval / test / predict pass.

    Args:
        cfg: parsed YAML config (dict). `run.mode` may be eval / test / predict.
        logger: already-configured logger
        ckpt_dir: working directory for this run (used e.g. for logs)
        progress_cb: optional callable invoked with per-batch progress info
                     when GUI needs a progress bar. CLI passes None.
    """
    mode = str(cfg.get("run", {}).get("mode", "eval")).lower()

    data_cfg = cfg.get("data", {})
    val_csv = data_cfg.get("val_csv", None)
    test_csv = data_cfg.get("test_csv", None)

    val_df = pd.read_csv(val_csv) if val_csv else None
    test_df = pd.read_csv(test_csv) if test_csv else None

    eval_df = test_df if test_df is not None else val_df
    if eval_df is None:
        raise ValueError("For eval/test/predict mode, test_csv or val_csv must be specified.")

    logger.info(f"Eval df: {len(eval_df)} rows")

    model, device, optimizer = build(cfg, logger=logger)

    eval_loader = make_loader(eval_df, cfg, mode=mode)

    ckpt_path = resolve_ckpt_path_for_mode(cfg, mode=mode)
    if not ckpt_path:
        raise ValueError("For eval/test/predict, set eval.load_pth (or run.load_pth).")
    load_checkpoint_into_model(model, ckpt_path, device, logger, strict=True)

    trainer = Trainer(
        cfg=cfg,
        model=model,
        device=device,
        optimizer=optimizer,
        logger=logger,
        ckpt_dir=ckpt_dir,
    )

    save_pred = bool(cfg.get("eval", {}).get("save_predictions", False))
    pred_dir = cfg.get("eval", {}).get("pred_dir", None)

    trainer.eval(eval_loader, save_predictions=save_pred, pred_dir=pred_dir)
