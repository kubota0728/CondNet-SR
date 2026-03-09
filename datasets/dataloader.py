# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 16:02:08 2026

@author: kubota
"""

# datasets/dataloader.py
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset as TorchDataset

class CondDataset(TorchDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int = 256,
        augmentation: bool = True,
        contrast_range=(0.8, 1.2),
        brightness_range=(-30, 30),
        augmentation_prob: float = 0.2,
        # df列名が環境で違う場合に差し替えられるようにしておく
        col_img1: str = "t1_img",
        col_img2: str = "t2_img",
        col_label: str = "label",
        col_mask: str = "t2mask",
        col_id: str = "IXI_ID",
        col_slice: str = "slice",
    ):
        self.df = df.reset_index(drop=True)
        self.image_size = int(image_size)

        self.img1path_list = self.df[col_img1].values
        self.img2path_list = self.df[col_img2].values
        self.labelpath_list = self.df[col_label].values
        self.t2mask_list    = self.df[col_mask].values
        
        # 追加：集約キー
        if col_id not in self.df.columns:
            raise KeyError(f"dfに {col_id} 列がありません")
        self.id_list = self.df[col_id].astype(str).values
        self.slice_list = self.df[col_slice].astype(int).values

        self.augmentation = bool(augmentation)
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
        self.augmentation_prob = float(augmentation_prob)

        # 0〜14 → 導電率(S/m) の直接マッピング（あなたの旧実装を合成して作成）
        # 元ラベル画像は「値/18」で 0..14 になる前提
        self.label_to_cond = {
            0: 0.0,
            1: 0.315,
            2: 0.036,
            3: 0.009,
            4: 0.0585,
            5: 0.9,
            6: 0.225,
            7: 0.018,
            8: 0.045,
            9: 0.0315,
            10: 0.153,
            11: 0.045,
            12: 0.675,
            13: 0.0315,
            14: 0.0806,
        }

    def apply_augmentation(self, image_u8: np.ndarray) -> np.ndarray:
        # contrast
        contrast = np.random.uniform(*self.contrast_range)
        image = image_u8.astype(np.float32) * float(contrast)

        # brightness
        b0, b1 = self.brightness_range
        brightness = np.random.randint(int(b0), int(b1) + 1)
        image = image + float(brightness)

        # clip
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def _read_gray_u8(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"読み込み失敗: {path}")
        if img.shape[:2] != (self.image_size, self.image_size):
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        return img

    def _read_mask01(self, path: str) -> np.ndarray:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"読み込み失敗: {path}")
        if mask.shape[:2] != (self.image_size, self.image_size):
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        return (mask > 0).astype(np.float32)

    def _read_label14(self, path: str) -> np.ndarray:
        # ラベル画像（0,18,36,... のような値）を想定
        lab = np.asarray(Image.open(path))
        if lab.ndim == 3:
            # 万一RGBなら先頭チャンネルを使用（必要なら適宜変更）
            lab = lab[..., 0]

        # まず nearest でサイズを合わせる（混色防止）
        if lab.shape[:2] != (self.image_size, self.image_size):
            lab = cv2.resize(lab, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # /18 で 0..14 に戻す（丸め誤差対策でround）
        lab14 = np.rint(lab.astype(np.float32) / 18.0).astype(np.int64)

        # 範囲外はクリップ
        lab14 = np.clip(lab14, 0, 14)
        return lab14

    def _label14_to_cond(self, lab14: np.ndarray) -> np.ndarray:
        # ベクトル化して 0..14 → cond へ
        # dict.get のdefaultも入れておく（念のため）
        vget = np.vectorize(lambda x: self.label_to_cond.get(int(x), 0.0), otypes=[np.float32])
        cond = vget(lab14).astype(np.float32)
        return cond

    def __getitem__(self, index: int):
        img1 = self._read_gray_u8(self.img1path_list[index])
        img2 = self._read_gray_u8(self.img2path_list[index])
        mask = self._read_mask01(self.t2mask_list[index])

        lab14 = self._read_label14(self.labelpath_list[index])
        cond  = self._label14_to_cond(lab14)  # (H,W) float32

        # 強度DAは入力だけ（教師condは変更しない）
        if self.augmentation and (np.random.rand() < self.augmentation_prob):
            img1 = self.apply_augmentation(img1)
            img2 = self.apply_augmentation(img2)

        # tensor化
        img1_t = torch.from_numpy((img1 / 255.0).astype(np.float32)).unsqueeze(0)  # (1,H,W)
        img2_t = torch.from_numpy((img2 / 255.0).astype(np.float32)).unsqueeze(0)  # (1,H,W)
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)            # (1,H,W)
        cond_t = torch.from_numpy(cond).unsqueeze(0)                               # (1,H,W)
        lab14_t = torch.from_numpy(lab14.astype(np.int64))  # (H,W) int64

        return {
            "img1": img1_t,
            "img2": img2_t,
            "mask": mask_t,
            "label": cond_t,   # ←学習側が "label" 前提ならこのまま。嫌なら "cond" に変更。
            "lab14": lab14_t,     # 1〜14評価用マスク
            "ixi_id": self.id_list[index],
            "slice": int(self.slice_list[index]),
        }

    def __len__(self):
        return len(self.img1path_list)