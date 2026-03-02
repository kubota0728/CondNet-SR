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
from typing import Dict, Optional, Any, Mapping, Tuple

import torch
import torch.nn as nn

from losses import build_loss


@dataclass
class EpochResult:
    loss: float
    seconds: float


class Trainer:
    """
    Trainer with explicit functions:
      - train(loader)
      - validate(loader)
      - eval(loader)
    Supports:
      - checkpoint saving: last/best
      - history tracking
      - optional frozen-transformer eval fixing (if model exposes method)
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        model: nn.Module | Dict[str, nn.Module],
        device: torch.device,
        optimizer: torch.optim.Optimizer | Dict[str, torch.optim.Optimizer],
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

        self.loss_fn = criterion if criterion is not None else build_loss(self.cfg)
        self._log(f"loss: {self.loss_fn.__class__.__name__}")

        self.history = {"train_loss": [], "val_loss": []}

        train_cfg = cfg.get("train", {})
        self.monitor = train_cfg.get("monitor", "val_loss")
        self.monitor_mode = train_cfg.get("monitor_mode", "min")
        self.best_score = None

        if bool(train_cfg.get("detect_anomaly", False)):
            torch.autograd.set_detect_anomaly(True)

        self.use_amp = bool(train_cfg.get("amp", False))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.log_every = int(train_cfg.get("log_every", 0))  # 0 means auto

    def _log(self, msg: str):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    # ----------------------------
    # model/optimizer helpers
    # ----------------------------
    def _models(self) -> Dict[str, nn.Module]:
        if isinstance(self.model, dict):
            return self.model
        return {"model": self.model}

    def _set_train_mode(self):
        for m in self._models().values():
            m.train()

    def _set_eval_mode(self):
        for m in self._models().values():
            m.eval()

    def _zero_grad(self):
        if isinstance(self.optimizer, dict):
            for opt in self.optimizer.values():
                opt.zero_grad(set_to_none=True)
        else:
            self.optimizer.zero_grad(set_to_none=True)

    def _optimizer_step(self):
        if isinstance(self.optimizer, dict):
            self.optimizer["ctnet"].step()
            if bool(self.cfg.get("train", {}).get("train_original", True)):
                self.optimizer["original"].step()
        else:
            self.optimizer.step()

    def _scaler_step(self):
        if isinstance(self.optimizer, dict):
            self.scaler.step(self.optimizer["ctnet"])
            if bool(self.cfg.get("train", {}).get("train_original", True)):
                self.scaler.step(self.optimizer["original"])
        else:
            self.scaler.step(self.optimizer)

    def _maybe_fix_frozen_transformers(self):
        train_cfg = self.cfg.get("train", {})
        if not bool(train_cfg.get("keep_frozen_transformer_eval", True)):
            return

        def _fix_one(m: nn.Module):
            if hasattr(m, "set_frozen_transformer_eval"):
                try:
                    m.set_frozen_transformer_eval()
                except Exception:
                    pass

        for m in self._models().values():
            _fix_one(m)

    # ----------------------------
    # batch/loss helpers
    # ----------------------------
    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    def _require_keys(self, batch: Mapping[str, Any], keys: list[str]):
        missing = [k for k in keys if k not in batch]
        if missing:
            raise KeyError(f"Batch missing keys: {missing}. Available keys: {list(batch.keys())}")

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor, batch: Optional[dict] = None) -> torch.Tensor:
        try:
            return self.loss_fn(pred, target, batch)
        except TypeError:
            return self.loss_fn(pred, target)

    def _monitor_better(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.monitor_mode == "min":
            return score < self.best_score
        return score > self.best_score

    # ----------------------------
    # helpers for UNet dict outputs
    # ----------------------------
    def _detach_unet_outputs(self, out_old: Any) -> Any:
        """
        UNet_2D returns dict {"t6","t7","t8","t9","t"} (Tensor).
        train_original=False のとき、ctnetへ渡す入力から勾配を止めるために
        dict中身をdetachする。
        """
        if isinstance(out_old, dict):
            out2 = {}
            for k, v in out_old.items():
                if torch.is_tensor(v):
                    out2[k] = v.detach()
                else:
                    out2[k] = v
            return out2
        # 万が一 tensor なら従来通り
        if torch.is_tensor(out_old):
            return out_old.detach()
        return out_old

    def _cpu_clone_outputs(self, out_old: Any) -> Any:
        """
        save_predictions 用。
        out_oldがdictでもtorch.saveできるように cpu() に落とす。
        """
        if isinstance(out_old, dict):
            out2 = {}
            for k, v in out_old.items():
                if torch.is_tensor(v):
                    out2[k] = v.detach().cpu()
                else:
                    out2[k] = v
            return out2
        if torch.is_tensor(out_old):
            return out_old.detach().cpu()
        return out_old

    # ----------------------------
    # forward helper (dict/単体 両対応)
    # ----------------------------
    def _forward(self, img1: torch.Tensor, img2: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Optional[Any]]:
        """
        Returns:
          pred: 最終出力（loss計算対象） Tensor [B,1,H,W] 想定
          out_old: originalの出力（dict運用時のみ） dict {"t6","t7","t8","t9","t"}
        """
        if isinstance(self.model, dict):
            out_old = self.model["original"](img1, img2, mask)

            if not bool(self.cfg.get("train", {}).get("train_original", True)):
                out_old = self._detach_unet_outputs(out_old)

            out = self.model["ctnet"](out_old)
            return out, out_old

        pred = self.model(img1, img2, mask)
        return pred, None

    # ----------------------------
    # checkpoint helpers (DP差分を吸収して保存/ロード安定化)
    # ----------------------------
    def _state_dict_of(self, m: nn.Module) -> dict:
        core = m.module if isinstance(m, nn.DataParallel) else m
        return core.state_dict()

    def _load_state_dict_into(self, m: nn.Module, sd: dict, strict: bool = True):
        core = m.module if isinstance(m, nn.DataParallel) else m

        # どちらでも読めるように module. を吸収
        def strip_module_prefix_if_needed(state_dict: dict) -> dict:
            if not any(k.startswith("module.") for k in state_dict.keys()):
                return state_dict
            return {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        # coreがDPでないのに sd が module. 付きなら剥ぐ
        # coreがDPである場合は coreはmoduleなので剥いでOK（core側はmoduleキー無しを期待）
        sd2 = strip_module_prefix_if_needed(sd)
        core.load_state_dict(sd2, strict=strict)

    def _save_checkpoint(self, name: str, epoch: int, extra: Optional[dict] = None):
        path = os.path.join(self.ckpt_dir, f"{name}.pth")

        if isinstance(self.model, dict):
            model_sd = {k: self._state_dict_of(v) for k, v in self.model.items()}
        else:
            model_sd = self._state_dict_of(self.model)

        if isinstance(self.optimizer, dict):
            opt_sd = {k: v.state_dict() for k, v in self.optimizer.items()}
        else:
            opt_sd = self.optimizer.state_dict()

        payload = {
            "epoch": epoch,
            "model": model_sd,
            "optimizer": opt_sd,
            "best_score": self.best_score,
            "cfg": self.cfg,
        }
        if extra:
            payload.update(extra)

        torch.save(payload, path)

    def load_for_eval_from_cfg(self):
        # 1) eval.load_pth を優先
        eval_cfg = self.cfg.get("eval", {})
        ckpt_path = str(eval_cfg.get("load_pth", "")).strip()
    
        # 2) 無ければ従来互換で run.load_pth を見る
        if not ckpt_path:
            run_cfg = self.cfg.get("run", {})
            ckpt_path = str(run_cfg.get("load_pth", "")).strip()
    
        if not ckpt_path:
            raise ValueError("eval.load_pth (or run.load_pth) is empty. Please set path to checkpoint.")
    
        ckpt = torch.load(ckpt_path, map_location=self.device)
    
        # trainer保存形式: {"model": ...}
        sd = ckpt["model"] if isinstance(ckpt, dict) and ("model" in ckpt) else ckpt
    
        if isinstance(self.model, dict):
            for k in ("original", "ctnet"):
                if k not in self.model:
                    raise KeyError(f"model dict must have key '{k}'")
                if isinstance(sd, dict) and k not in sd:
                    raise KeyError(f"checkpoint model state_dict missing key '{k}'")
            self.model["original"].load_state_dict(sd["original"], strict=True)
            self.model["ctnet"].load_state_dict(sd["ctnet"], strict=True)
        else:
            self.model.load_state_dict(sd, strict=True)
    
        self._log(f"Loaded eval weights: {ckpt_path}")

    def _save_training_artifacts(self):
        import pickle
        import matplotlib.pyplot as plt

        os.makedirs(self.ckpt_dir, exist_ok=True)

        train_loss = self.history.get("train_loss", [])
        val_loss = self.history.get("val_loss", [])

        if len(train_loss) > 0:
            plt.figure()
            plt.plot(train_loss, label="train")
            if len(val_loss) > 0:
                plt.plot(val_loss, label="val")

            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend()
            plt.grid(True)

            png_path = os.path.join(self.ckpt_dir, "loss_curve.png")
            plt.savefig(png_path, dpi=200, bbox_inches="tight")
            plt.close()

            self._log(f"Loss curve saved to: {png_path}")

        pkl_path = os.path.join(self.ckpt_dir, "history.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(self.history, f)

        self._log(f"History saved to: {pkl_path}")

    # ----------------------------
    # loops
    # ----------------------------
    def train(self, train_loader) -> EpochResult:
        t0 = time.time()
        self._set_train_mode()
        self._maybe_fix_frozen_transformers()

        running = 0.0
        n_batches = 0

        log_every = self.log_every
        if log_every <= 0:
            log_every = max(1, len(train_loader) // 10)

        autocast_enabled = self.use_amp and (self.device.type == "cuda")

        for i, batch in enumerate(train_loader, start=1):
            batch = self._to_device(batch)
            self._require_keys(batch, ["img1", "img2", "label", "mask"])

            img1 = batch["img1"]
            img2 = batch["img2"]
            label = batch["label"]
            mask = batch["mask"]

            self._zero_grad()

            if autocast_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred, _out_old = self._forward(img1, img2, mask)
                    loss = self._compute_loss(pred, label, batch)

                self.scaler.scale(loss).backward()
                self._scaler_step()
                self.scaler.update()
            else:
                pred, _out_old = self._forward(img1, img2, mask)
                loss = self._compute_loss(pred, label, batch)

                loss.backward()
                self._optimizer_step()

            running += float(loss.item())
            n_batches += 1

            if i % log_every == 0:
                self._log(f"[train] iter {i}/{len(train_loader)} | loss={running/n_batches:.6f}")

        epoch_loss = running / max(1, n_batches)
        sec = time.time() - t0
        self.history["train_loss"].append(epoch_loss)
        return EpochResult(loss=epoch_loss, seconds=sec)

    @torch.no_grad()
    def validate(self, val_loader) -> EpochResult:
        t0 = time.time()
        self._set_eval_mode()
        self._maybe_fix_frozen_transformers()

        running = 0.0
        n_batches = 0

        for batch in val_loader:
            batch = self._to_device(batch)
            self._require_keys(batch, ["img1", "img2", "label", "mask"])

            img1 = batch["img1"]
            img2 = batch["img2"]
            label = batch["label"]
            mask = batch["mask"]

            pred, _out_old = self._forward(img1, img2, mask)
            loss = self._compute_loss(pred, label, batch)

            running += float(loss.item())
            n_batches += 1

        epoch_loss = running / max(1, n_batches)
        sec = time.time() - t0
        self.history["val_loss"].append(epoch_loss)
        return EpochResult(loss=epoch_loss, seconds=sec)

    @torch.no_grad()
    def eval(self, loader, save_predictions: bool = False, pred_dir: Optional[str] = None) -> Dict[str, Any]:
        self._set_eval_mode()
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
            self._require_keys(batch, ["img1", "img2", "mask"])

            img1 = batch["img1"]
            img2 = batch["img2"]
            mask = batch["mask"]
            label = batch.get("label", None)

            pred, out_old = self._forward(img1, img2, mask)

            if label is None:
                has_label = False
            else:
                loss = self._compute_loss(pred, label, batch)
                running += float(loss.item())
                n_batches += 1

            if save_predictions:
                if out_old is not None:
                    p = {
                        "original": self._cpu_clone_outputs(out_old),  # dict対応
                        "ctnet": pred.detach().cpu(),
                    }
                    torch.save(p, os.path.join(pred_dir, f"pred_{idx:06d}.pt"))
                else:
                    torch.save(pred.detach().cpu(), os.path.join(pred_dir, f"pred_{idx:06d}.pt"))
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
        epochs = int(self.cfg.get("train", {}).get("epochs", 50))
        save_every = int(self.cfg.get("train", {}).get("save_every", 1))

        self._log(f"device: {self.device}")
        self._log(f"epochs: {epochs} | amp: {self.use_amp}")
        self._log(f"ckpt_dir: {self.ckpt_dir}")

        try:
            for epoch in range(1, epochs + 1):
                tr = self.train(train_loader)
                va = self.validate(val_loader)

                self._log(
                    f"[epoch {epoch:03d}] "
                    f"train_loss={tr.loss:.6f} ({tr.seconds:.1f}s) | "
                    f"val_loss={va.loss:.6f} ({va.seconds:.1f}s)"
                )

                self._save_checkpoint("last", epoch)

                score = va.loss
                if self._monitor_better(score):
                    self.best_score = score
                    self._save_checkpoint("best", epoch)
                    self._log(f"✅ best updated: {self.best_score:.6f}")

                if save_every > 0 and (epoch % save_every == 0):
                    self._save_checkpoint(f"epoch_{epoch:03d}", epoch)

        finally:
            self._save_training_artifacts()