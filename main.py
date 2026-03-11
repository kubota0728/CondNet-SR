# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 15:52:56 2026

@author: kubota
"""

# main.py
import argparse
import os
import yaml
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.logger import setup_logger, close_logger
from datasets.dataloader import CondDataset
from engine.builder import build
from engine.trainer import Trainer

import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torch.cuda.nccl"
)

def _prepare_output_dirs(cfg, timestamp: str):
    out_cfg = cfg.get("output", {})
    ckpt_root = out_cfg.get("ckpt_root", "checkpoints")

    ckpt_dir = os.path.join(ckpt_root, timestamp)
    os.makedirs(ckpt_dir, exist_ok=True)

    cfg.setdefault("output", {})
    cfg["output"]["ckpt_dir"] = ckpt_dir

    return ckpt_dir


def _make_loader(df, cfg, mode: str):
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

    ds = CondDataset(df, image_size=image_size,augmentation=augmentation)
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


def _resolve_ckpt_path_for_mode(cfg, mode: str) -> str:
    """
    eval系: eval.load_pth 優先、無ければ run.load_pth
    train系: run.load_pth（resume用）
    """
    run_cfg = cfg.get("run", {})
    eval_cfg = cfg.get("eval", {})

    if mode in ("eval", "test", "predict"):
        p = str(eval_cfg.get("load_pth", "")).strip()
        if p:
            return p
        return str(run_cfg.get("load_pth", "")).strip()

    # train resume
    return str(run_cfg.get("load_pth", "")).strip()


def _load_checkpoint_into_model(model, ckpt_path: str, device, logger, strict: bool = True):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    logger, log_file, timestamp = setup_logger(logs_root="logs", enable_print_redirect=False)
    ckpt_dir = _prepare_output_dirs(cfg, timestamp)
    
    logger.info("===== Program Started =====")
    logger.info(f"config: {args.config}")
    logger.info(f"log_file: {log_file}")
    logger.info(f"ckpt_dir: {ckpt_dir}")
    
    logger.info("===== Config =====")
    logger.info("\n" + yaml.dump(cfg, sort_keys=False, allow_unicode=True))

    try:
        mode = str(cfg.get("run", {}).get("mode", "train")).lower()

        data_cfg = cfg.get("data", {})
        train_csv = data_cfg.get("train_csv", None)
        val_csv = data_cfg.get("val_csv", None)
        test_csv = data_cfg.get("test_csv", None)

        train_df = pd.read_csv(train_csv) if train_csv else None
        val_df = pd.read_csv(val_csv) if val_csv else None
        test_df = pd.read_csv(test_csv) if test_csv else None

        # まずモデル構築（train/eval共通）
        model, device, optimizer = build(cfg, logger=logger)

        # -------------------------
        # TRAIN
        # -------------------------
        if mode == "train":
            if train_df is None or val_df is None:
                raise ValueError("For train mode, train_csv and val_csv are required.")

            logger.info(f"Train df: {len(train_df)} rows")
            logger.info(f"Val df: {len(val_df)} rows")

            train_loader = _make_loader(train_df, cfg, mode="train")
            val_loader = _make_loader(val_df, cfg, mode="val")

            # resume が必要ならロード（run.load_pth）
            ckpt_path = _resolve_ckpt_path_for_mode(cfg, mode="train")
            if ckpt_path:
                _load_checkpoint_into_model(model, ckpt_path, device, logger, strict=True)

            trainer = Trainer(
                cfg=cfg,
                model=model,
                device=device,
                optimizer=optimizer,
                logger=logger,
                ckpt_dir=ckpt_dir,
            )
            trainer.fit(train_loader, val_loader)

        # -------------------------
        # EVAL / TEST / PREDICT
        # -------------------------
        elif mode in ("eval", "test", "predict"):
            eval_df = test_df if test_df is not None else val_df
            if eval_df is None:
                raise ValueError("For eval/test/predict mode, test_csv or val_csv must be specified.")

            logger.info(f"Eval df: {len(eval_df)} rows")
            eval_loader = _make_loader(eval_df, cfg, mode=mode)

            ckpt_path = _resolve_ckpt_path_for_mode(cfg, mode=mode)
            if not ckpt_path:
                raise ValueError("For eval/test/predict, set eval.load_pth (or run.load_pth).")
            _load_checkpoint_into_model(model, ckpt_path, device, logger, strict=True)

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

        else:
            raise ValueError(f"Unknown run.mode: {mode}")

        logger.info("===== Program Finished =====")

    except Exception as e:
        logger.exception(f"Error occurred: {e}")
        raise
    finally:
        close_logger(logger)
        print("Logger closed.")


if __name__ == "__main__":
    main()