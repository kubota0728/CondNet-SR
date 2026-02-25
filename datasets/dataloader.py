# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 16:02:08 2026

@author: kubota
"""

# datasets/dataloader.py
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class CondDataset(Dataset):
    def __init__(self, df, image_size=256):
        self.img1path_list = df.img1path.values
        self.img2path_list = df.img2path.values
        self.labelpath_list = df.labelpath.values
        self.image_size = image_size

        # ラベルマッピング
        self.label_mapping = {
            0: 0,
            1: 9,
            2: 9,
            3: 1,
            4: 2,
            5: 3,
            6: 9,
            7: 9,
            8: 4,
            9: 5,
            10: 9,
            11: 6,
            12: 7,
            13: 8,
            14: 9
        }

        self.replacement_values = {
            0: 0,
            1: 0.009,
            2: 0.0585,
            3: 0.9,
            4: 0.045,
            5: 0.0315,
            6: 0.045,
            7: 0.675,
            8: 0.0315,
            9: 0.1494
        }

    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        if h != self.image_size or w != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size))
        img = img / 255.0
        img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        return img

    def _load_label(self, path):
        label = Image.open(path)
        label = np.asarray(label)
        label = label / 18  # 元のラベルに戻す

        label = np.vectorize(self.label_mapping.get)(label).astype(float)

        for original_value, new_value in self.replacement_values.items():
            label[label == original_value] = new_value

        h, w = label.shape
        if h != self.image_size or w != self.image_size:
            label = cv2.resize(label, (self.image_size, self.image_size))

        label = torch.from_numpy(label.astype(np.float32)).unsqueeze(0)
        return label

    def __getitem__(self, index):
        img1 = self._load_image(self.img1path_list[index])
        img2 = self._load_image(self.img2path_list[index])
        label = self._load_label(self.labelpath_list[index])

        return {
            "img1": img1,
            "img2": img2,
            "label": label
        }

    def __len__(self):
        return len(self.img1path_list)