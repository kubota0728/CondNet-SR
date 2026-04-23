# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 16:12:38 2026

@author: kubota
"""

# engine/trainer.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, Mapping, Tuple

import torch
import torch.nn as nn

from losses import build_loss

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class EpochResult:
    loss: float
    seconds: float
    terms: dict = field(default_factory=dict)


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
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        self.log_every = int(train_cfg.get("log_every", 0))  # 0 means auto
        
        self.last_loss_terms = {}
        
        self.monitor_slide = int(cfg["val"].get("monitor_slide", 150))
        

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

    def _compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        batch: Optional[dict] = None
    ) -> torch.Tensor:
    
        loss = self.loss_fn(pred, target, batch=batch)
    
        # loss内訳を保存
        terms = getattr(self.loss_fn, "last_terms", None)
        if isinstance(terms, dict):
            self.last_loss_terms = dict(terms)
        else:
            self.last_loss_terms = {}
    
        return loss

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
            self._require_keys(batch, ["img1", "img2", "label", "mask", "lab14"])

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
    def validate(
        self,
        val_loader,
        epoch,
        preview_cases: Optional[List[Tuple[str, int]]] = None,
        preview_cb: Optional[Callable[[int, List[dict]], None]] = None,
    ) -> EpochResult:
        """
        Args:
            val_loader : 検証データローダ
            epoch      : エポック番号
            preview_cases : GUI プレビュー対象の (ixi_id, slice) タプル列。
                            None/空で挙動変化なし。
            preview_cb    : epoch 終端で preview サンプル群を受け取るコールバック。
                            None ならスキップ。

        挙動互換:
            preview_cases / preview_cb を渡さない限り、既存の monitor_slide 処理
            も loss 計算も変更されていません。
        """
        t0 = time.time()
        self._set_eval_mode()
        self._maybe_fix_frozen_transformers()

        running = 0.0
        n_batches = 0

        term_sums = {}
        has_terms = False

        # 監視用スライス番号（yaml: val.monitor_slide）
        monitor_slide = int(getattr(self, "monitor_slide", 150))

        # 監視用の1枚だけ保持
        vis_sample = None

        # 必要なら導電率スケールを元に戻す
        SCALE_BACK = 2.0 / 0.9

        # GUI プレビュー用: (pid, slice) -> 収集した sample dict
        preview_key_set = set()
        if preview_cases:
            for pid, sl in preview_cases:
                preview_key_set.add((str(pid), int(sl)))
        preview_collected: dict = {}

        with torch.no_grad():
            for batch in val_loader:
                batch = self._to_device(batch)
                self._require_keys(batch, ["img1", "img2", "label", "mask", "lab14", "slice"])

                img1 = batch["img1"]
                img2 = batch["img2"]
                label = batch["label"]
                mask = batch["mask"]
                lab14 = batch["lab14"]
                slices = batch["slice"]
                ixi_ids = batch.get("ixi_id", None)

                pred, _out_old = self._forward(img1, img2, mask)
                loss = self._compute_loss(pred, label, batch)

                running += float(loss.item())
                n_batches += 1

                # loss項の集計
                terms = getattr(self, "last_loss_terms", None)
                if isinstance(terms, dict) and len(terms) > 0:
                    has_terms = True
                    for k, v in terms.items():
                        term_sums[k] = term_sums.get(k, 0.0) + float(v)

                # 監視用スライスを最初の1枚だけ取得
                if isinstance(slices, torch.Tensor):
                    slices_list = slices.detach().cpu().tolist()
                else:
                    slices_list = list(slices)

                if vis_sample is None:
                    B = int(pred.shape[0])
                    for i in range(B):
                        if int(slices_list[i]) == monitor_slide:
                            vis_sample = {
                                "t1": img1[i, 0].detach().cpu().numpy(),
                                "t2": img2[i, 0].detach().cpu().numpy(),
                                "gt": label[i, 0].detach().cpu().numpy().astype(np.float32) * SCALE_BACK,
                                "pred": pred[i, 0].detach().cpu().numpy().astype(np.float32) * SCALE_BACK,
                                "lab14": lab14[i].detach().cpu().numpy(),
                                "slice": int(slices_list[i]),
                            }
                            break

                # GUI プレビュー: 指定された (pid, slice) を収集
                if preview_key_set and ixi_ids is not None:
                    if isinstance(ixi_ids, torch.Tensor):
                        ixi_ids_list = [str(x) for x in ixi_ids.detach().cpu().tolist()]
                    else:
                        ixi_ids_list = [str(x) for x in ixi_ids]
                    B = int(pred.shape[0])
                    for i in range(B):
                        key = (ixi_ids_list[i], int(slices_list[i]))
                        if key in preview_key_set and key not in preview_collected:
                            preview_collected[key] = {
                                "pid": key[0],
                                "slice": key[1],
                                "t1": img1[i, 0].detach().cpu().numpy().astype(np.float32),
                                "t2": img2[i, 0].detach().cpu().numpy().astype(np.float32),
                                "gt": label[i, 0].detach().cpu().numpy().astype(np.float32) * SCALE_BACK,
                                "pred": pred[i, 0].detach().cpu().numpy().astype(np.float32) * SCALE_BACK,
                            }

        epoch_loss = running / max(1, n_batches)
        sec = time.time() - t0
        self.history["val_loss"].append(epoch_loss)

        avg_terms = {}
        if has_terms and n_batches > 0:
            avg_terms = {k: v / n_batches for k, v in term_sums.items()}

        # 監視用画像を1枚だけ表示（既存挙動そのまま）
        if vis_sample is not None:
            self._show_val_monitor_slice(
                t1_2d=vis_sample["t1"],
                t2_2d=vis_sample["t2"],
                gt_2d=vis_sample["gt"],
                pr_2d=vis_sample["pred"],
                lb_2d=vis_sample["lab14"],
                slice_id=vis_sample["slice"],
                epoch=epoch,
            )
        else:
            self._log(f"[val] monitor_slide={monitor_slide} のスライスは見つかりませんでした。")

        # GUI プレビューへ送出 (preview_cases の順序で整える)
        if preview_cb is not None and preview_key_set:
            ordered = []
            for pid, sl in preview_cases:
                key = (str(pid), int(sl))
                if key in preview_collected:
                    ordered.append(preview_collected[key])
            try:
                preview_cb(epoch, ordered)
            except Exception as e:
                self._log(f"[preview_cb] error ignored: {e}")

        return EpochResult(loss=epoch_loss, seconds=sec, terms=avg_terms)

    def _save_pred_nii_3d(self, pred_3d_zhw: np.ndarray, save_path: str):
        """
        pred_3d_zhw: (Z, H, W) float32
        niiは(H, W, Z)として保存（一般的な見た目のため）
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vol_hwz = np.transpose(pred_3d_zhw, (1, 2, 0)).astype(np.float32)  # (H,W,Z)
        affine = np.eye(4, dtype=np.float32)
        nib.save(nib.Nifti1Image(vol_hwz, affine), save_path)
    
    def _show_mid_slice(self, t1_zhw, t2_zhw, gt_zhw, pred_zhw, pid: str):

        z = int(pred_zhw.shape[0] // 2)
    
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    
        axes[0,0].imshow(t1_zhw[z], cmap="gray", interpolation="nearest")
        axes[0,0].set_title("T1", fontsize=16)
    
        axes[0,1].imshow(t2_zhw[z], cmap="gray", interpolation="nearest")
        axes[0,1].set_title("T2", fontsize=16)
    
        axes[1,0].imshow(gt_zhw[z],
                         cmap="jet",
                         vmin=0.0,
                         vmax=2.2,
                         interpolation="nearest")
        axes[1,0].set_title("GT", fontsize=16)
    
        axes[1,1].imshow(pred_zhw[z],
                         cmap="jet",
                         vmin=0.0,
                         vmax=2.2,
                         interpolation="nearest")
        axes[1,1].set_title("Pred", fontsize=16)
    
        for ax in axes.ravel():
            ax.axis("off")
    
        fig.suptitle(f"{pid} (z={z})", fontsize=18)
    
        plt.tight_layout()
        plt.show()
    
    def _show_val_monitor_slice(self, t1_2d, t2_2d, gt_2d, pr_2d, lb_2d, slice_id, epoch):
        fig, axes = plt.subplots(
            1, 6,
            figsize=(20, 4),
            gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 0.05]}
        )
    
        axes[0].imshow(t1_2d, cmap="gray", interpolation="nearest")
        axes[0].set_title("T1")
        axes[0].axis("off")
    
        axes[1].imshow(t2_2d, cmap="gray", interpolation="nearest")
        axes[1].set_title("T2")
        axes[1].axis("off")
    
        axes[2].imshow(lb_2d, vmin=0, vmax=14, interpolation="nearest")
        axes[2].set_title("lab14")
        axes[2].axis("off")
    
        im = axes[3].imshow(gt_2d, cmap="jet", vmin=0, vmax=2.2, interpolation="nearest")
        axes[3].set_title("GT")
        axes[3].axis("off")
    
        axes[4].imshow(pr_2d, cmap="jet", vmin=0, vmax=2.2, interpolation="nearest")
        axes[4].set_title("Pred")
        axes[4].axis("off")
    
        fig.colorbar(im, cax=axes[5])
    
        plt.suptitle(f"Epoch {epoch} | Validation Monitor Slice = {slice_id}")
        plt.tight_layout()
    
        save_dir = os.path.join(self.ckpt_dir, "val_monitor")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"epoch_{epoch:03d}_slice_{slice_id}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    def _compute_metrics_case(self, pred_zhw, gt_zhw, lab14_zhw):
        """
        pred/gt: (Z,H,W) float
        lab14: (Z,H,W) int 0..14
        指標は表示用にdictで返す
        """
        out = {}
    
        m_all = (lab14_zhw >= 1) & (lab14_zhw <= 14)
        out["MAE_all"] = float(np.mean(np.abs(pred_zhw[m_all] - gt_zhw[m_all]))) if m_all.any() else float("nan")
    
        for r in range(1, 15):
            m = (lab14_zhw == r)
            if not m.any():
                continue
            pv = pred_zhw[m]
            gv = gt_zhw[m]
            out[f"Label{r:02d}_MAE"] = float(np.mean(np.abs(pv - gv)))
            out[f"Label{r:02d}_mean"] = float(np.mean(pv))
            out[f"Label{r:02d}_std"]  = float(np.std(pv, ddof=0))
        return out
        

    @torch.no_grad()
    def eval(self, loader, save_predictions=False, pred_dir=None):

        SCALE_BACK = 2.0 / 0.9
    
        self._set_eval_mode()
        self._maybe_fix_frozen_transformers()
    
        if save_predictions:
            if pred_dir is None:
                pred_dir = os.path.join(self.ckpt_dir, "predictions_nii")
            os.makedirs(pred_dir, exist_ok=True)
    
        # buffers[pid] = list of (slice, t1, t2, gt, pred, lab14)
        buffers = {}
    
        for idx, batch in enumerate(loader):
            batch = self._to_device(batch)
            self._require_keys(batch, ["img1", "img2", "mask", "label", "lab14", "ixi_id", "slice"])
    
            img1 = batch["img1"]
            img2 = batch["img2"]
            mask = batch["mask"]
            gt   = batch["label"]   # (B,1,H,W)
            lab14 = batch["lab14"]  # (B,H,W)
            ixi_ids = batch["ixi_id"]
            slices  = batch["slice"]
    
            pred, _ = self._forward(img1, img2, mask)  # (B,1,H,W)
    
            B = int(pred.shape[0])
    
            # ixi_ids, slices を list 化（collate対策）
            if not isinstance(ixi_ids, (list, tuple)):
                ixi_ids = list(ixi_ids)
            if not isinstance(slices, (list, tuple)):
                slices = list(slices)
    
            for i in range(B):
                pid = str(ixi_ids[i])
                s   = int(slices[i])
    
                t1_2d = img1[i, 0].detach().cpu().numpy()
                t2_2d = img2[i, 0].detach().cpu().numpy()
    
                # ★導電率スケール復元（pred/gtともに）
                gt_2d = gt[i, 0].detach().cpu().numpy().astype(np.float32) * SCALE_BACK
                pr_2d = pred[i, 0].detach().cpu().numpy().astype(np.float32) * SCALE_BACK
    
                lb_2d = lab14[i].detach().cpu().numpy().astype(np.int16)
    
                buffers.setdefault(pid, []).append((s, t1_2d, t2_2d, gt_2d, pr_2d, lb_2d))
    
        # ===== 全症例集約用（ラベル別 mean/std/MAE、全体MAE） =====
        global_acc = {r: {"n": 0, "sum": 0.0, "sumsq": 0.0, "abs_err": 0.0} for r in range(1, 15)}
        global_all = {"n": 0, "abs_err": 0.0}
    
        # ★IDごとに処理
        for pid, items in buffers.items():
            items.sort(key=lambda x: x[0])  # slice順に並べる
            n_slices = len(items)
            self._log(f"[eval] IXI_ID={pid} | slices={n_slices}")
    
            # stackして3D化: (Z,H,W)
            t1_zhw = np.stack([x[1] for x in items], axis=0).astype(np.float32)
            t2_zhw = np.stack([x[2] for x in items], axis=0).astype(np.float32)
            gt_zhw = np.stack([x[3] for x in items], axis=0).astype(np.float32)
            pr_zhw = np.stack([x[4] for x in items], axis=0).astype(np.float32)
            lb_zhw = np.stack([x[5] for x in items], axis=0).astype(np.int16)
    
            # ---- 症例ごとのログは必要最小限（細かい出力はしない） ----
            m_all = (lb_zhw >= 1) & (lb_zhw <= 14)
            if m_all.any():
                mae_case_all = float(np.mean(np.abs(pr_zhw[m_all] - gt_zhw[m_all])))
                self._log(f"  MAE_all={mae_case_all:.5f}")
            else:
                self._log("  MAE_all=nan")
    
            # ---- z//2の断面を表示（保存しない） ----
            self._show_mid_slice(t1_zhw, t2_zhw, gt_zhw, pr_zhw, pid)
    
            # ---- predのみnii保存（症例1ファイル） ----
            if save_predictions:
                save_path = os.path.join(pred_dir, f"{pid}_pred.nii.gz")
                self._save_pred_nii_3d(pr_zhw, save_path)
    
            # ---- 全症例集約（label別 mean/std/MAE と 全体MAE）----
            if m_all.any():
                ae = np.abs(pr_zhw[m_all] - gt_zhw[m_all])
                global_all["n"] += int(ae.size)
                global_all["abs_err"] += float(ae.sum())
    
            for r in range(1, 15):
                m = (lb_zhw == r)
                if not m.any():
                    continue
                pv = pr_zhw[m]
                gv = gt_zhw[m]
                n = int(pv.size)
                global_acc[r]["n"] += n
                global_acc[r]["sum"] += float(pv.sum())
                global_acc[r]["sumsq"] += float((pv * pv).sum())
                global_acc[r]["abs_err"] += float(np.abs(pv - gv).sum())
    
        # ===== 全症例での最終出力（小数点第5位まで） =====
        if global_all["n"] > 0:
            mae_all = global_all["abs_err"] / global_all["n"]
            self._log(f"[eval] GLOBAL | MAE_all={mae_all:.5f}")
        else:
            self._log("[eval] GLOBAL | MAE_all=nan")
    
        for r in range(1, 15):
            n = global_acc[r]["n"]
            if n == 0:
                continue
            mean = global_acc[r]["sum"] / n
            var = global_acc[r]["sumsq"] / n - mean * mean
            if var < 0.0:
                var = 0.0
            std = float(np.sqrt(var))
            mae = global_acc[r]["abs_err"] / n
            self._log(f"[eval] GLOBAL | Label: {r:02d} | MAE: {mae:.5f} | mean±SD: {mean:.5f}±{std:.5f}")
    
        return {"done": True, "n_ids": len(buffers)}

    def fit(
        self,
        train_loader,
        val_loader,
        progress_cb: Optional[Callable[[dict], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
        preview_cb: Optional[Callable[[int, List[dict]], None]] = None,
        preview_cases: Optional[List[Tuple[str, int]]] = None,
    ):
        """
        学習ループ本体。

        Args:
            train_loader, val_loader : データローダ
            progress_cb  : 各エポック終了時に dict で進捗情報を受け取るコールバック。
                           GUI の進捗バー・ETA・loss 曲線・ベスト表示に使う。
                           None で挙動変化なし (CLI)。
            stop_check   : 各エポック後に呼び出される zero-arg コールバック。
                           True を返すと finally を経由して学習を終了する。
            preview_cb   : validate() 末尾で 3 ケース程度のプレビューを受け取る。
            preview_cases: プレビュー対象 (ixi_id, slice) のリスト。
        """
        epochs = int(self.cfg.get("train", {}).get("epochs", 50))
        save_every = int(self.cfg.get("train", {}).get("save_every", 1))

        self._log(f"device: {self.device}")
        self._log(f"epochs: {epochs} | amp: {self.use_amp}")
        self._log(f"ckpt_dir: {self.ckpt_dir}")

        fit_start = time.time()

        try:
            for epoch in range(1, epochs + 1):

                self._log("")
                self._log("=" * 60)
                self._log(f"Epoch {epoch}/{epochs}")
                self._log("=" * 60)

                tr = self.train(train_loader)
                va = self.validate(
                    val_loader,
                    epoch,
                    preview_cases=preview_cases,
                    preview_cb=preview_cb,
                )

                # validation loss の内訳がある場合だけ一緒に表示
                if hasattr(va, "terms") and isinstance(va.terms, dict) and len(va.terms) > 0:
                    terms_str = ", ".join(
                        [f"{k}={v:.6f}" for k, v in va.terms.items()]
                    )
                    self._log(
                        f"[epoch {epoch:03d}] "
                        f"train_loss={tr.loss:.6f} ({tr.seconds:.1f}s) | "
                        f"val_loss={va.loss:.6f} "
                        f"[{terms_str}] "
                        f"({va.seconds:.1f}s)"
                    )
                else:
                    self._log(
                        f"[epoch {epoch:03d}] "
                        f"train_loss={tr.loss:.6f} ({tr.seconds:.1f}s) | "
                        f"val_loss={va.loss:.6f} ({va.seconds:.1f}s)"
                    )

                self._save_checkpoint("last", epoch)

                score = va.loss
                is_best = self._monitor_better(score)
                if is_best:
                    self.best_score = score
                    self._save_checkpoint("best", epoch)
                    self._log(f"✅ best updated: {self.best_score:.6f}")

                if save_every > 0 and (epoch % save_every == 0):
                    self._save_checkpoint(f"epoch_{epoch:03d}", epoch)

                # GUI 向け進捗コールバック
                if progress_cb is not None:
                    try:
                        progress_cb({
                            "epoch": epoch,
                            "total_epochs": epochs,
                            "train_loss": float(tr.loss),
                            "val_loss": float(va.loss),
                            "val_terms": dict(va.terms) if isinstance(va.terms, dict) else {},
                            "train_seconds": float(tr.seconds),
                            "val_seconds": float(va.seconds),
                            "elapsed_seconds": float(time.time() - fit_start),
                            "best_loss": (float(self.best_score) if self.best_score is not None else None),
                            "is_best": bool(is_best),
                        })
                    except Exception as e:
                        self._log(f"[progress_cb] error ignored: {e}")

                # 停止要求 (GUI の Stop ボタン)
                if stop_check is not None:
                    try:
                        if stop_check():
                            self._log("学習停止が要求されました。現在のエポック終了時点で中断します。")
                            break
                    except Exception as e:
                        self._log(f"[stop_check] error ignored: {e}")

        finally:
            self._save_training_artifacts()