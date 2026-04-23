# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 16:59:29 2026

@author: kubota
"""

# seg_pretrain.py
# 1-file segmentation pretraining script with progress printing

# ----------------------------
# Imports
# ----------------------------
import os
import sys
import pickle
import datetime
from dataclasses import dataclass
from typing import Tuple, Optional

import pandas as pd
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

import segmentation_models_pytorch as smp


# ----------------------------
# Tee logger
# ----------------------------
class Tee:
    def __init__(self, filename: str):
        self.file = open(filename, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, message: str):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    # paths
    train_csv: str = "D:/kubota/Data/Model10test/filepath/train_df_20260226.csv"
    val_csv: str   = "D:/kubota/Data/Model10test/filepath/val_df_20260226.csv"
    test_csv: str  = "D:/kubota/Data/Model10test/filepath/test_df_20260226.csv"

    out_dir: str = "D:/kubota/Data/Model10test/seg_result/"
    log_filename: str = "output.log"
    history_pkl: str = "history.pkl"

    # data
    image_size: int = 256
    num_classes: int = 15
    num_workers: int = 0

    # train
    batch_size: int = 32
    epochs: int = 50
    lr: float = 1e-3

    # augmentation
    augmentation: bool = True
    augmentation_prob: float = 0.5
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    brightness_range: Tuple[int, int] = (-30, 30)

    # progress printing
    train_print_splits: int = 10  # 10分割で途中表示
    # val途中表示をしたいならここを True に
    val_print_progress: bool = False
    val_print_splits: int = 1  # Trueにするなら 5 など

    # reproducibility (必要なら)
    seed: int = 42


# ----------------------------
# Utilities
# ----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 速度優先の設定（再現性優先なら調整）
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def get_device_and_wrap(model: nn.Module):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using {torch.cuda.device_count()} GPU(s)!")
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
    model.to(device)
    return device, model


def safe_interval(num_batches: int, splits: int) -> int:
    # num_batches を splits 分割したい。最低1にする。
    if splits <= 0:
        return max(1, num_batches)
    return max(1, num_batches // splits)


# ----------------------------
# Dataset
# ----------------------------
class Dataset(BaseDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        num_classes: int,
        image_size: int = 256,
        augmentation: bool = True,
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        brightness_range: Tuple[int, int] = (-30, 30),
        augmentation_prob: float = 0.5,
    ):
        self.img1path_list = df["t1_img"].values
        self.img2path_list = df["t2_img"].values
        self.labelpath_list = df["label"].values
        self.t2mask_list   = df["t2mask"].values

        self.num_classes = num_classes
        self.image_size = image_size

        self.augmentation = augmentation
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
        self.augmentation_prob = augmentation_prob

    def apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        # contrast
        contrast = np.random.uniform(*self.contrast_range)
        image = image.astype(np.float32) * contrast

        # brightness
        brightness = np.random.randint(self.brightness_range[0], self.brightness_range[1] + 1)
        image = image + float(brightness)

        # clip
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def _read_gray(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"読み込み失敗: {path}")
        if img.shape[:2] != (self.image_size, self.image_size):
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        if self.augmentation and (np.random.rand() < self.augmentation_prob):
            img = self.apply_augmentation(img)
        return img

    def _read_label(self, path: str) -> np.ndarray:
        lab = np.asarray(Image.open(path))
        # ラベルは必ず nearest（混色防止）
        if lab.shape[:2] != (self.image_size, self.image_size):
            lab = cv2.resize(lab, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # 元実装踏襲：/18 でクラスID化
        lab = (lab / 18).astype(np.int64)

        # one_hotで落ちないようにクリップ
        lab = np.clip(lab, 0, self.num_classes - 1)
        return lab

    def __getitem__(self, i):
        img1 = self._read_gray(self.img1path_list[i])
        img2 = self._read_gray(self.img2path_list[i])
        lab  = self._read_label(self.labelpath_list[i])
    
        # ---- mask 読み込み ----
        mask = cv2.imread(self.t2mask_list[i], cv2.IMREAD_GRAYSCALE)
        if mask.shape[:2] != (self.image_size, self.image_size):
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
    
        # 0/255 → 0/1 に変換
        mask = (mask > 0).astype(np.float32)
    
        # ---- tensor化 ----
        img1 = torch.from_numpy((img1 / 255.0).astype(np.float32)).unsqueeze(0)
        img2 = torch.from_numpy((img2 / 255.0).astype(np.float32)).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)   # (1,H,W)
    
        lab_t = torch.from_numpy(lab).long()
        lab_oh = F.one_hot(lab_t, num_classes=self.num_classes).permute(2,0,1).float()
    
        return {
            "img1": img1,
            "img2": img2,
            "mask": mask,
            "label": lab_oh
        }

    def __len__(self):
        return len(self.img1path_list)


def build_loaders(cfg: Config):
    train_df = pd.read_csv(cfg.train_csv)
    val_df   = pd.read_csv(cfg.val_csv)
    test_df  = pd.read_csv(cfg.test_csv)

    train_dataset = Dataset(
        train_df,
        num_classes=cfg.num_classes,
        image_size=cfg.image_size,
        augmentation=cfg.augmentation,
        contrast_range=cfg.contrast_range,
        brightness_range=cfg.brightness_range,
        augmentation_prob=cfg.augmentation_prob,
    )
    val_dataset = Dataset(
        val_df,
        num_classes=cfg.num_classes,
        image_size=cfg.image_size,
        augmentation=False,
    )
    test_dataset = Dataset(
        test_df,
        num_classes=cfg.num_classes,
        image_size=cfg.image_size,
        augmentation=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    return train_df, val_df, test_df, train_loader, val_loader, test_loader


# ----------------------------
# Model (U-Net + Transformer)
# ----------------------------
class TwoConvBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.rl = nn.ReLU()
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.rl(self.bn1(self.conv1(x)))
        x = self.rl(self.bn2(self.conv2(x)))
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.bn1(x)
        x = self.conv(x)
        x = self.bn2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_size: int, heads: int, dropout: float = 0.1):
        super().__init__()
        # (B, L, E) をそのまま扱えるよう batch_first=True
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: (B, L, E)
        attn_out, _ = self.attention(x, x, x, need_weights=False)
        x = self.dropout(self.norm1(attn_out + x))
        ff = self.feed_forward(x)
        x = self.dropout(self.norm2(ff + x))
        return x


class UNet_2D(nn.Module):
    def __init__(self, num_classes: int, transformer_count: int = 6):
        super().__init__()
        self.TCB1 = TwoConvBlock(3, 32, 32)
        self.TCB2 = TwoConvBlock(32, 64, 64)
        self.TCB3 = TwoConvBlock(64, 128, 128)
        self.TCB4 = TwoConvBlock(128, 256, 256)
        self.TCB5 = TwoConvBlock(256, 512, 512)
        self.TCB6 = TwoConvBlock(512, 256, 256)
        self.TCB7 = TwoConvBlock(256, 128, 128)
        self.TCB8 = TwoConvBlock(128, 64, 64)
        self.TCB9 = TwoConvBlock(64, 32, 32)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.UC1 = UpConv(512, 256)
        self.UC2 = UpConv(256, 128)
        self.UC3 = UpConv(128, 64)
        self.UC4 = UpConv(64, 32)

        self.conv1 = nn.Conv2d(32, num_classes, kernel_size=1)

        self.Trans = nn.ModuleList([TransformerBlock(embed_size=512, heads=4) for _ in range(transformer_count)])

    def forward(self, a, b, mask):
        x = torch.cat([a, b, mask], dim=1)

        x = self.TCB1(x); x1 = x; x = self.maxpool(x)
        x = self.TCB2(x); x2 = x; x = self.maxpool(x)
        x = self.TCB3(x); x3 = x; x = self.maxpool(x)
        x = self.TCB4(x); x4 = x; x = self.maxpool(x)

        x = self.TCB5(x)  # (B,512,Hb,Wb)

        # Transformer on flattened spatial tokens
        B, C, Hb, Wb = x.shape
        x = x.view(B, C, Hb * Wb).permute(0, 2, 1)  # (B, L, E)

        for tr in self.Trans:
            x = tr(x)

        x = x.permute(0, 2, 1).contiguous().view(B, C, Hb, Wb)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim=1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.TCB9(x)

        x = self.conv1(x)  # logits
        return x


def build_model(cfg: Config) -> nn.Module:
    return UNet_2D(num_classes=cfg.num_classes, transformer_count=6)


# ----------------------------
# Loss (same idea as your original)
# ----------------------------
def build_criterion():
    tversky = smp.losses.TverskyLoss(mode="multilabel", log_loss=False)
    bce     = smp.losses.SoftBCEWithLogitsLoss()

    def criterion(pred, target, channels=None):
        if channels is not None:
            pred = pred[:, channels, :, :]
            target = target[:, channels, :, :]
        return 0.5 * bce(pred, target) + 0.5 * tversky(pred, target)

    return criterion


# ----------------------------
# Train / Validate
# ----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion,
    device: torch.device,
    epoch: int,
    cfg: Config,
    history: dict,
    channels_to_use: Optional[list] = None,
):
    model.train()
    num_batches = len(loader)
    log_int = safe_interval(num_batches, cfg.train_print_splits)

    running = 0.0
    running_n = 0

    for i, data in enumerate(loader, start=1):
        inputs1 = data["img1"].to(device)
        inputs2 = data["img2"].to(device)
        labels  = data["label"].to(device)
        mask = data["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs1, inputs2, mask)
        loss = criterion(outputs, labels, channels_to_use)

        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        history["train_loss"].append(loss_val)

        running += loss_val
        running_n += 1

        # 途中経過表示（10分割相当、0除算はsafe_intervalで回避）
        if (i % log_int) == 0 or (i == num_batches):
            print(f"epoch:{epoch}  index:{i}  train_loss:{running / max(1, running_n):.5f}")
            running = 0.0
            running_n = 0


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    epoch: int,
    cfg: Config,
    channels_to_use: Optional[list] = None,
) -> float:
    model.eval()
    core = model.module if isinstance(model, nn.DataParallel) else model 
    total = 0.0
    n = 0

    num_batches = len(loader)
    log_int = safe_interval(num_batches, cfg.val_print_splits) if cfg.val_print_progress else None
    running = 0.0
    running_n = 0

    with torch.no_grad():
        for i, data in enumerate(loader, start=1):
            inputs1 = data["img1"].to(device)
            inputs2 = data["img2"].to(device)
            labels  = data["label"].to(device)
            mask = data["mask"].to(device)

            outputs = core(inputs1, inputs2, mask)
            loss = criterion(outputs, labels, channels_to_use)

            loss_val = float(loss.item())
            total += loss_val
            n += 1

            if cfg.val_print_progress:
                running += loss_val
                running_n += 1
                if (i % log_int) == 0 or (i == num_batches):
                    print(f"epoch:{epoch}  index:{i}  val_loss(partial):{running / max(1, running_n):.5f}")
                    running = 0.0
                    running_n = 0

    val_loss = total / max(1, n)
    print(f"epoch:{epoch}  val_loss:{val_loss:.5f}")
    return val_loss


def save_checkpoint(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)


# ----------------------------
# Plot history
# ----------------------------
def plot_history(history: dict, epochs: int):
    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")

    if len(history["train_loss"]) > 0 and len(history["val_loss"]) > 0:
        step = max(1, len(history["train_loss"]) // epochs)
        xs = list(range(step, step * len(history["val_loss"]) + 1, step))
        xs = xs[:len(history["val_loss"])]
        plt.plot(xs, history["val_loss"], label="Validation Loss", marker="o")

    plt.xlabel("batch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
    plt.close()


# ----------------------------
# Main
# ----------------------------
def main():
    cfg = Config()
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)

    # Tee logging
    original_stdout = sys.stdout
    log_file = Tee(cfg.log_filename)
    sys.stdout = log_file

    try:
        train_df, val_df, test_df, train_loader, val_loader, test_loader = build_loaders(cfg)

        model = build_model(cfg)
        device, model = get_device_and_wrap(model)

        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        criterion = build_criterion()

        history = {"train_loss": [], "val_loss": []}

        dt_now = datetime.datetime.now()
        print("start training", dt_now)

        # 必要なら特定チャンネルだけでloss計算
        channels_to_use_all = None  # 例: [1] とか

        best_val = float("inf")
        best_path = os.path.join(cfg.out_dir, "best.pth")
        last_path = os.path.join(cfg.out_dir, "last.pth")

        for epoch in range(1, cfg.epochs + 1):
            dt_now = datetime.datetime.now()
            print(f"{epoch} epochの計算 {dt_now}")

            # train with progress prints
            train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch=epoch,
                cfg=cfg,
                history=history,
                channels_to_use=channels_to_use_all,
            )

            # validate (epoch mean)
            val_loss = validate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                epoch=epoch,
                cfg=cfg,
                channels_to_use=channels_to_use_all,
            )
            history["val_loss"].append(val_loss)

            # save each epoch (as your original)
            epoch_path = os.path.join(cfg.out_dir, f"train_{epoch}.pth")
            save_checkpoint(model, epoch_path)

            # also save last & best (追加：研究運用で便利)
            save_checkpoint(model, last_path)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(model, best_path)
                print(f"best updated: val_loss={best_val:.5f}")

        dt_now = datetime.datetime.now()
        print("finish training", dt_now)

        # save history
        with open(cfg.history_pkl, "wb") as f:
            pickle.dump(history, f)

        # plot
        plot_history(history, cfg.epochs)

    finally:
        # restore stdout
        sys.stdout = original_stdout
        log_file.close()

    print("計算終了")
    # この後に test 推論を足したいなら、predict関数を追加してここで呼べばOK


if __name__ == "__main__":
    main()