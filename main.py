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


def _prepare_output_dirs(cfg):
    run_cfg = cfg.get("run", {})
    exp_name = run_cfg.get("exp_name", "exp01")

    out_cfg = cfg.get("output", {})
    logs_root = out_cfg.get("logs_root", "logs")
    ckpt_root = out_cfg.get("ckpt_root", "checkpoints")

    logs_dir = os.path.join(logs_root, exp_name)
    ckpt_dir = os.path.join(ckpt_root, exp_name)

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    cfg.setdefault("output", {})
    cfg["output"]["logs_dir"] = logs_dir
    cfg["output"]["ckpt_dir"] = ckpt_dir
    cfg["run"]["exp_name"] = exp_name

    return logs_dir, ckpt_dir


def _make_loader(df, cfg, mode: str):
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    val_cfg = cfg.get("val", {})

    image_size = int(data_cfg.get("image_size", 256))
    batch_size = int(data_cfg.get("batch_size", 32))

    if mode == "train":
        shuffle = True
        num_workers = int(train_cfg.get("num_workers", 0))
        bs = batch_size
    elif mode == "val":
        shuffle = False
        num_workers = int(val_cfg.get("num_workers", 0))
        bs = batch_size
    elif mode == "test":
        shuffle = False
        num_workers = int(val_cfg.get("num_workers", 0))
        bs = 1
    else:
        raise ValueError(f"Unknown loader mode: {mode}")

    ds = CondDataset(df, image_size=image_size)
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logs_dir, ckpt_dir = _prepare_output_dirs(cfg)

    logger = setup_logger(logs_dir=logs_dir, enable_print_redirect=False)
    logger.info("===== Program Started =====")
    logger.info(f"config: {args.config}")
    logger.info(f"logs_dir: {logs_dir}")
    logger.info(f"ckpt_dir: {ckpt_dir}")

    try:
        mode = cfg.get("run", {}).get("mode", "train")

        data_cfg = cfg.get("data", {})
        train_csv = data_cfg.get("train_csv", None)
        val_csv = data_cfg.get("val_csv", None)
        test_csv = data_cfg.get("test_csv", None)

        train_df = pd.read_csv(train_csv) if train_csv else None
        val_df = pd.read_csv(val_csv) if val_csv else None
        test_df = pd.read_csv(test_csv) if test_csv else None

        # =========================
        # TRAIN MODE
        # =========================
        if mode == "train":
            if train_df is None or val_df is None:
                raise ValueError("For train mode, train_csv and val_csv are required.")

            logger.info(f"Train df: {len(train_df)} rows")
            logger.info(f"Val df: {len(val_df)} rows")

            train_loader = _make_loader(train_df, cfg, mode="train")
            val_loader = _make_loader(val_df, cfg, mode="val")

            model, device, optimizer = build(cfg, logger=logger)

            # 🔥 ここで load_pth が指定されていれば読み込む（resume用）
            load_pth = cfg.get("run", {}).get("load_pth", None)
            if load_pth:
                if not os.path.isfile(load_pth):
                    raise FileNotFoundError(f"Specified pth not found: {load_pth}")

                logger.info(f"Loading model weights from: {load_pth}")
                ckpt = torch.load(load_pth, map_location=device)
                state_dict = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt

                if isinstance(model, torch.nn.DataParallel):
                    model.module.load_state_dict(state_dict, strict=True)
                else:
                    model.load_state_dict(state_dict, strict=True)

                logger.info("Model weights loaded successfully.")

            trainer = Trainer(
                cfg=cfg,
                model=model,
                device=device,
                optimizer=optimizer,
                logger=logger,
                ckpt_dir=ckpt_dir,
            )

            trainer.fit(train_loader, val_loader)

        # =========================
        # EVAL / TEST MODE
        # =========================
        elif mode in ("eval", "predict", "test"):
            eval_df = test_df if test_df is not None else val_df
            if eval_df is None:
                raise ValueError("For eval mode, test_csv or val_csv must be specified.")

            logger.info(f"Eval df: {len(eval_df)} rows")
            eval_loader = _make_loader(eval_df, cfg, mode="test")

            model, device, optimizer = build(cfg, logger=logger)

            # 🔥 ここが一番重要：指定pthを必ず読む
            load_pth = cfg.get("run", {}).get("load_pth", None)
            if not load_pth:
                raise ValueError("In eval mode, run.load_pth must be specified.")

            if not os.path.isfile(load_pth):
                raise FileNotFoundError(f"Specified pth not found: {load_pth}")

            logger.info(f"Loading model weights from: {load_pth}")
            ckpt = torch.load(load_pth, map_location=device)
            state_dict = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt

            if isinstance(model, torch.nn.DataParallel):
                model.module.load_state_dict(state_dict, strict=True)
            else:
                model.load_state_dict(state_dict, strict=True)

            logger.info("Model weights loaded successfully.")

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