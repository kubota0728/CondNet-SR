# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 16:12:38 2026

@author: kubota
"""

# engine/trainer.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Any

import torch
import torch.nn as nn


@dataclass
class EpochResult:
    loss: float
    seconds: float


class Trainer:
    """
    Trainer with explicit functions:
      - train(loader)
      - validate(loader)
      - eval(loader)  # = evaluate
    Supports:
      - checkpoint saving: last/best
      - history tracking
      - optional frozen-transformer eval fixing (if model exposes method)
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        model: nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        logger=None,
        ckpt_dir: str = "checkpoints",
        criterion: Optional[nn.Module] = None,
    ):
        self.cfg = cfg
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.logger = logger
        self.ckpt_dir = ckpt_dir

        os.makedirs(self.ckpt_dir, exist_ok=True)

        # default loss = L1
        self.criterion = criterion if criterion is not None else nn.L1Loss()

        # history
        self.history = {
            "train_loss": [],
            "val_loss": [],
        }

        # checkpoint monitor
        train_cfg = cfg.get("train", {})
        self.monitor = train_cfg.get("monitor", "val_loss")          # only "val_loss" supported here
        self.monitor_mode = train_cfg.get("monitor_mode", "min")     # "min" or "max"
        self.best_score = None

        # anomaly detection
        if bool(train_cfg.get("detect_anomaly", False)):
            torch.autograd.set_detect_anomaly(True)

        # mixed precision (optional)
        self.use_amp = bool(train_cfg.get("amp", False))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # logging frequency
        self.log_every = int(train_cfg.get("log_every", 0))  # 0 means "auto" (10 times/epoch)

    def _log(self, msg: str):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    def _maybe_fix_frozen_transformers(self):
        """
        If you froze transformer params, you may want them to stay in eval mode.
        This mirrors your old behavior.
        Works if model has method: set_frozen_transformer_eval()
        """
        train_cfg = self.cfg.get("train", {})
        if not bool(train_cfg.get("keep_frozen_transformer_eval", True)):
            return
        if hasattr(self.model, "set_frozen_transformer_eval"):
            try:
                self.model.set_frozen_transformer_eval()
            except Exception:
                pass

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.criterion(pred, target)

    def _monitor_better(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.monitor_mode == "min":
            return score < self.best_score
        return score > self.best_score

    def _save_checkpoint(self, name: str, epoch: int, extra: Optional[dict] = None):
        path = os.path.join(self.ckpt_dir, f"{name}.pth")
        payload = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_score": self.best_score,
            "cfg": self.cfg,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
    
    def _save_loss_curve(self):
        """
        Save loss curve to checkpoints directory.
        """
        import os
        import matplotlib.pyplot as plt
    
        train_loss = self.history.get("train_loss", [])
        val_loss = self.history.get("val_loss", [])
    
        if len(train_loss) == 0:
            return
    
        plt.figure()
        plt.plot(train_loss, label="train")
        if len(val_loss) > 0:
            plt.plot(val_loss, label="val")
    
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.grid(True)
    
        out_path = os.path.join(self.ckpt_dir, "loss_curve.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
    
        self._log(f"Loss curve saved to: {out_path}")

    def train(self, train_loader) -> EpochResult:
        """
        One epoch training.
        Returns EpochResult(loss, seconds).
        """
        t0 = time.time()
        self.model.train()
        self._maybe_fix_frozen_transformers()

        running = 0.0
        n_batches = 0

        # auto log every ~10% if not set
        log_every = self.log_every
        if log_every <= 0:
            log_every = max(1, len(train_loader) // 10)

        for i, batch in enumerate(train_loader, start=1):
            batch = self._to_device(batch)
            img1 = batch["img1"]
            img2 = batch["img2"]
            label = batch["label"]

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    pred = self.model(img1, img2)
                    loss = self._compute_loss(pred, label)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(img1, img2)
                loss = self._compute_loss(pred, label)
                loss.backward()
                self.optimizer.step()

            running += float(loss.item())
            n_batches += 1

            if i % log_every == 0:
                self._log(f"[train] iter {i}/{len(train_loader)} | loss={running/n_batches:.6f}")

        epoch_loss = running / max(1, n_batches)
        sec = time.time() - t0
        self.history["train_loss"].append(epoch_loss)
        
        # training finished
        self._save_loss_curve()

        return EpochResult(loss=epoch_loss, seconds=sec)

    @torch.no_grad()
    def validate(self, val_loader) -> EpochResult:
        """
        One epoch validation.
        Returns EpochResult(loss, seconds).
        """
        t0 = time.time()
        self.model.eval()
        self._maybe_fix_frozen_transformers()

        running = 0.0
        n_batches = 0

        for batch in val_loader:
            batch = self._to_device(batch)
            img1 = batch["img1"]
            img2 = batch["img2"]
            label = batch["label"]

            pred = self.model(img1, img2)
            loss = self._compute_loss(pred, label)

            running += float(loss.item())
            n_batches += 1

        epoch_loss = running / max(1, n_batches)
        sec = time.time() - t0
        self.history["val_loss"].append(epoch_loss)
        return EpochResult(loss=epoch_loss, seconds=sec)

    @torch.no_grad()
    def eval(self, loader, save_predictions: bool = False, pred_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluation / prediction.
        - Computes mean loss (if labels exist)
        - Optionally saves predictions as .pt per sample (simple and safe).
        """
        self.model.eval()
        self._maybe_fix_frozen_transformers()

        if save_predictions:
            if pred_dir is None:
                pred_dir = os.path.join(self.ckpt_dir, "predictions")
            os.makedirs(pred_dir, exist_ok=True)

        has_label = True
        running = 0.0
        n_batches = 0
        saved = 0

        for idx, batch in enumerate(loader):
            batch = self._to_device(batch)
            img1 = batch["img1"]
            img2 = batch["img2"]
            label = batch.get("label", None)

            pred = self.model(img1, img2)

            if label is None:
                has_label = False
            else:
                loss = self._compute_loss(pred, label)
                running += float(loss.item())
                n_batches += 1

            if save_predictions:
                # Save prediction tensor (CPU) per batch index
                p = pred.detach().cpu()
                torch.save(p, os.path.join(pred_dir, f"pred_{idx:06d}.pt"))
                saved += 1

        result = {
            "has_label": has_label,
            "mean_loss": (running / max(1, n_batches)) if has_label else None,
            "saved_predictions": saved if save_predictions else 0,
            "pred_dir": pred_dir if save_predictions else None,
        }
        self._log(f"[eval] done | mean_loss={result['mean_loss']} | saved={result['saved_predictions']}")
        return result

    def fit(self, train_loader, val_loader):
        """
        Full training loop (epochs) that calls train/validate and saves checkpoints.
        """
        epochs = int(self.cfg.get("train", {}).get("epochs", 50))
        save_every = int(self.cfg.get("train", {}).get("save_every", 1))

        self._log(f"device: {self.device}")
        self._log(f"epochs: {epochs} | amp: {self.use_amp}")
        self._log(f"ckpt_dir: {self.ckpt_dir}")

        for epoch in range(1, epochs + 1):
            tr = self.train(train_loader)
            va = self.validate(val_loader)

            self._log(
                f"[epoch {epoch:03d}] "
                f"train_loss={tr.loss:.6f} ({tr.seconds:.1f}s) | "
                f"val_loss={va.loss:.6f} ({va.seconds:.1f}s)"
            )

            # last checkpoint
            self._save_checkpoint("last", epoch)

            # best checkpoint (monitor val_loss only here)
            score = va.loss
            if self._monitor_better(score):
                self.best_score = score
                self._save_checkpoint("best", epoch)
                self._log(f"✅ best updated: {self.best_score:.6f}")

            # optional per-epoch save
            if save_every > 0 and (epoch % save_every == 0):
                self._save_checkpoint(f"epoch_{epoch:03d}", epoch)