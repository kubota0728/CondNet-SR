# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 11:40:19 2026

@author: kubota
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MAELoss(nn.Module):
    def forward(self, pred, target, batch=None):
        return torch.mean(torch.abs(pred - target))


class MixedLogMAELoss(nn.Module):
    """
    0.95*MAE + 0.05*logMAE
    logMAE = mean(|log(pred) - log(target)|)
    """
    def __init__(self, eps: float = 1e-6, w_mae: float = 0.95, w_log: float = 0.05):
        super().__init__()
        self.eps = float(eps)
        self.w_mae = float(w_mae)
        self.w_log = float(w_log)

    def forward(self, pred, target, batch=None):
        # pred/target: (B,C,H,W) or (B,1,H,W)
        pred_safe = torch.clamp(pred, min=self.eps)
        target_safe = torch.clamp(target, min=self.eps)

        mae = torch.mean(torch.abs(pred - target))
        logmae = torch.mean(torch.abs(torch.log(pred_safe) - torch.log(target_safe)))

        return self.w_mae * mae + self.w_log * logmae


class CondNetCSLoss(nn.Module):
    """
    CondNet-CS loss for 2D slice training.

    必要な batch:
        batch["img1"]   : T1,   [B,1,H,W] or [B,H,W]
        batch["img2"]   : T2,   [B,1,H,W] or [B,H,W]
        batch["lab14"]  : 元の 0~14 ラベル, [B,1,H,W] or [B,H,W]

    ※ batch["mask"] があっても、この loss では使用しない。
      mask は入力側で T2 欠損補完などに使うためのものであり、
      loss 計算には含めない。

    forward:
        pred   : 予測導電率 [B,1,H,W] or [B,H,W]
        target : 導電率教師(label) [B,1,H,W] or [B,H,W]
    """

    def __init__(
        self,
        lambda_stat=1.0,
        lambda_rank=1.0,
        lambda_smooth=1.0,
        w_mean=0.5,
        w_std=0.5,
        rank_pairs=100,
        alpha=0.5,
        eps=1e-6,
        full_pair_threshold=20,
    ):
        super().__init__()

        self.lambda_stat = float(lambda_stat)
        self.lambda_rank = float(lambda_rank)
        self.lambda_smooth = float(lambda_smooth)

        self.w_mean = float(w_mean)
        self.w_std = float(w_std)

        self.rank_pairs = int(rank_pairs)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.full_pair_threshold = int(full_pair_threshold)

        # rank / std の対象部位
        self.rank_std_labels = [5, 8, 13]  # CSF, GM, WM

        # CV target
        self.target_cv = {
            5: 0.077,   # CSF
            8: 0.227,   # GM
            13: 0.348,  # WM
        }

        self.last_terms = {}
        self.last_debug = {}
        
        self.mean_loss = MAELoss()
        # self.mean_loss = MixedLogMAELoss()

    def forward(self, pred, target, batch=None):
        if batch is None:
            raise ValueError("CondNetCSLoss requires batch.")
        if "lab14" not in batch:
            raise ValueError("batch['lab14'] is required.")
        if "img1" not in batch or "img2" not in batch:
            raise ValueError("batch['img1'] and batch['img2'] are required.")

        pred = self._to_b1hw(pred).float()
        label = self._to_b1hw(target).float()
        label14 = self._to_b1hw(batch["lab14"]).long()
        t1 = self._to_b1hw(batch["img1"]).float()
        t2 = self._to_b1hw(batch["img2"]).float()

        # PixelMap
        pixel_map = (1.0 - self.alpha) * (1.0 - t1) + self.alpha * t2

        self.last_debug = {
            "mean_pixels": int(pred.numel()),
            "std_regions": 0,
            "rank_regions": 0,
            "smooth_regions": 0,
        }

        # MAE は純粋な画素ごとの MAE
        loss_mean = self.mean_loss(pred, label)

        # 重みが 0 の項は計算しない
        if self.lambda_stat != 0.0 and self.w_std != 0.0:
            loss_std = self._loss_std(pred, label14)
        else:
            loss_std = pred.sum() * 0.0

        if self.lambda_rank != 0.0:
            loss_rank = self._loss_rank(pred, pixel_map, label14)
        else:
            loss_rank = pred.sum() * 0.0

        if self.lambda_smooth != 0.0:
            loss_smooth = self._loss_smooth(pred, label14)
        else:
            loss_smooth = pred.sum() * 0.0

        loss_stat = self.w_mean * loss_mean + self.w_std * loss_std

        total = (
            self.lambda_stat * loss_stat
            + self.lambda_rank * loss_rank
            + self.lambda_smooth * loss_smooth
        )

        self.last_terms = {
            "loss_mean": float(loss_mean.detach().item()),
            "loss_std": float(loss_std.detach().item()),
            "loss_stat": float(loss_stat.detach().item()),
            "loss_rank": float(loss_rank.detach().item()),
            "loss_smooth": float(loss_smooth.detach().item()),
            "loss_total": float(total.detach().item()),
        }

        return total

    def _to_b1hw(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            x = x.unsqueeze(1)
        elif x.ndim == 4:
            if x.shape[1] > 1:
                x = x[:, :1, :, :]
        else:
            raise ValueError(f"Unsupported tensor shape: {x.shape}")
        return x

    def _binary_erode(self, region_mask, iterations=1):
        x = region_mask.float()
        for _ in range(iterations):
            x = 1.0 - F.max_pool2d(1.0 - x, kernel_size=3, stride=1, padding=1)
            x = (x > 0.5).float()
        return x > 0.5

    def _get_core_masks(self, region_mask):
        core1 = self._binary_erode(region_mask, iterations=1)
        core2 = self._binary_erode(region_mask, iterations=2)

        # 小領域で消えすぎるのを防ぐ
        if int(core1.sum().item()) < 2:
            core1 = region_mask.clone()
        if int(core2.sum().item()) < 3:
            core2 = core1.clone()

        return core1, core2

    def _loss_std(self, pred, label14):
        vals = []

        batch_size = pred.shape[0]
        for b in range(batch_size):
            for lab_id in self.rank_std_labels:
                region = (label14[b:b+1] == lab_id)
                if int(region.sum().item()) < 3:
                    continue

                _, core2 = self._get_core_masks(region)
                if int(core2.sum().item()) < 3:
                    continue

                x = pred[b:b+1][core2]
                mu = x.mean()
                std = x.std(unbiased=False)
                cv_pred = std / (torch.abs(mu) + self.eps)

                cv_tgt = pred.new_tensor(self.target_cv[lab_id])
                vals.append(torch.abs(cv_pred - cv_tgt))
                self.last_debug["std_regions"] += 1

        if len(vals) == 0:
            return pred.sum() * 0.0

        return torch.stack(vals).mean()

    def _sample_pairs(self, n, k, device, full_pair_threshold=20):
        if n < 2:
            return None, None

        if n <= full_pair_threshold:
            idx = torch.arange(n, device=device)
            ii, jj = torch.meshgrid(idx, idx, indexing="ij")
            valid = (ii < jj)  # 自己比較・重複除去
            ii = ii[valid].reshape(-1)
            jj = jj[valid].reshape(-1)
            if ii.numel() == 0:
                return None, None
            return ii, jj

        i = torch.randint(0, n, (k,), device=device)
        j = torch.randint(0, n, (k,), device=device)

        valid = (i != j)
        if int(valid.sum().item()) == 0:
            return None, None

        return i[valid], j[valid]

    def _loss_rank(self, pred, pixel_map, label14):
        vals = []

        batch_size = pred.shape[0]
        for b in range(batch_size):
            for lab_id in self.rank_std_labels:
                region = (label14[b:b+1] == lab_id)
                if int(region.sum().item()) < 2:
                    continue

                core1, _ = self._get_core_masks(region)
                if int(core1.sum().item()) < 2:
                    continue

                sigma = pred[b, 0][core1[0, 0]]
                inten = pixel_map[b, 0][core1[0, 0]]

                ii, jj = self._sample_pairs(
                    sigma.numel(),
                    self.rank_pairs,
                    pred.device,
                    full_pair_threshold=self.full_pair_threshold,
                )
                if ii is None:
                    continue

                sigma_i = sigma[ii]
                sigma_j = sigma[jj]
                inten_i = inten[ii]
                inten_j = inten[jj]

                s = torch.sign(inten_i - inten_j)
                valid = (s != 0)
                if int(valid.sum().item()) == 0:
                    continue

                sigma_i = sigma_i[valid]
                sigma_j = sigma_j[valid]
                s = s[valid]

                l = F.relu(-(sigma_i - sigma_j) * s)
                vals.append(l.mean())
                self.last_debug["rank_regions"] += 1

        if len(vals) == 0:
            return pred.sum() * 0.0

        return torch.stack(vals).mean()

    def _loss_smooth(self, pred, label14):
        vals = []

        batch_size = pred.shape[0]
        for b in range(batch_size):
            for lab_id in self.rank_std_labels:
                region = (label14[b:b+1] == lab_id)
                if int(region.sum().item()) < 3:
                    continue

                _, core2 = self._get_core_masks(region)
                if int(core2.sum().item()) < 3:
                    continue

                p = pred[b:b+1]
                c = core2

                dx = torch.abs(p[:, :, :, 1:] - p[:, :, :, :-1])
                mx = c[:, :, :, 1:] & c[:, :, :, :-1]

                dy = torch.abs(p[:, :, 1:, :] - p[:, :, :-1, :])
                my = c[:, :, 1:, :] & c[:, :, :-1, :]

                local_vals = []
                if int(mx.sum().item()) > 0:
                    local_vals.append(dx[mx].mean())
                if int(my.sum().item()) > 0:
                    local_vals.append(dy[my].mean())

                if len(local_vals) > 0:
                    vals.append(torch.stack(local_vals).mean())
                    self.last_debug["smooth_regions"] += 1

        if len(vals) == 0:
            return pred.sum() * 0.0

        return torch.stack(vals).mean()


def build_loss(cfg):
    name = cfg.get("loss", {}).get("name", "mae").lower()

    if name == "mae":
        return MAELoss()

    if name == "logmae":
        return MixedLogMAELoss()

    if name == "condnet_cs":
        loss_cfg = cfg.get("loss", {})
        return CondNetCSLoss(
            lambda_stat=loss_cfg.get("lambda_stat", 1.0),
            lambda_rank=loss_cfg.get("lambda_rank", 0.2),
            lambda_smooth=loss_cfg.get("lambda_smooth", 0.01),
            w_mean=loss_cfg.get("w_mean", 0.5),
            w_std=loss_cfg.get("w_std", 0.5),
            rank_pairs=loss_cfg.get("rank_pairs", 100),
            alpha=loss_cfg.get("alpha", 0.5),
            full_pair_threshold=loss_cfg.get("full_pair_threshold", 20),
        )

    raise ValueError(f"Unknown loss: {name}")