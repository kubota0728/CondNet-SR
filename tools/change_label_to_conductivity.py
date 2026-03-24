# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 10:47:22 2026

@author: kubota
"""

import numpy as np
import nibabel as nib


def label_to_conductivity(label_zyx: np.ndarray) -> np.ndarray:
    """
    ラベル (0–14) → 導電率に変換
    仮の値を設定
    """
    
    cond_map = {
        0: 0,
        1: 0.7,
        2: 0.08,
        3: 0.02,
        4: 0.13,
        5: 2.0,
        6: 0.5,
        7: 0.04,
        8: 0.1,
        9: 0.07,
        10: 0.34,
        11: 0.1,
        12: 1.5,
        13: 0.07,
        14: 0.18
    }

    cond = np.zeros_like(label_zyx, dtype=np.float32)

    for k, v in cond_map.items():
        cond[label_zyx == k] = v

    return cond


def rotate_label_to_match_cond(cond_zyx: np.ndarray) -> np.ndarray:
    """
    rotate_cond_to_match_label の逆変換
    """
    v = cond_zyx

    # transpose は self-inverse
    v = np.transpose(v, (1, 0, 2))

    return v


def convert_label_to_cond_nii(label_path, output_path):

    # NIfTI 読み込み
    nii = nib.load(label_path)
    label = nii.get_fdata().astype(np.int16)

    # ラベル → 導電率
    cond = label_to_conductivity(label)

    # 保存前に逆回転
    cond = rotate_label_to_match_cond(cond)

    # NIfTI 保存
    cond_nii = nib.Nifti1Image(cond, nii.affine, nii.header)

    nib.save(cond_nii, output_path)

    print("saved:", output_path)


# 使用例
pid=401
label_path = f"D:/kubota/Data/Model9/label14/IXI{pid:03d}_label_after.nii.gz"
output_path = f"D:/kubota/Data/Model10test/uniform_nii/{pid}_uniform.nii.gz"

convert_label_to_cond_nii(label_path, output_path)