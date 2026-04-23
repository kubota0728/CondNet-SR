# -*- coding: utf-8 -*-
"""
Export NIfTI volumes to per-slice PNG files and build the dataloader path CSV.

Step 2 of the preprocessing pipeline. Run after 01_stratified_split.py.

Prerequisites (done in upstream Model8 pipeline; NOT included in this repo):
  - T1 / T2 MRI and label map co-registered
  - T1 / T2 intensities 99-percentile normalized
  (The "_after" suffix in NIfTI filenames marks this processed state.)

Input:
  - Split CSV from Step 1 (set CSV_PATH below; re-run once per split)
  - NIfTI files named:
      <IMAGE_DIR>/IXI{id:03d}_T1_after.nii.gz
      <IMAGE_DIR>/IXI{id:03d}_T2_after.nii.gz
      <LABEL_DIR>/IXI{id:03d}_label_after.nii.gz

Output (per-case folders under OUT_DIR/<IXI_prefix>/):
  - img/   IXI{id:03d}_z{zzz}_T1.png,  _T2.png
  - label/ IXI{id:03d}_z{zzz}_label.png    (raw label x 18 for visibility;
                                            the dataloader divides by 18)
  - t2mask/IXI{id:03d}_z{zzz}_T2mask.png   (T2-missing mask: T1>T1_THRESH & T2==0)
  - <OUT_DIR>/paths_all_slices.csv         (path CSV consumed by CondDataset)

Note:
  normalize_volume_to_uint8() performs only a volume-wise min-max mapping to
  0-255 for PNG storage. It is NOT an intensity-correction step; 99-percentile
  normalization is assumed to be already done upstream.

Key settings (edit constants below):
  CSV_PATH, LABEL_DIR, IMAGE_DIR, OUT_DIR,
  T1_THRESH, T2_ZERO_EPS, MASK_SCALE_255.

See tools/preprocess/README.md for the full pipeline context and output
CSV column specification.

Created on Thu Feb 26 12:20:47 2026
@author: kubota
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import cv2


# =========================
# User settings
# =========================
CSV_PATH = r"D:\kubota\Data\Model10\split_out/test.csv"

LABEL_DIR = r"D:\kubota\Data\Model9\label14"
IMAGE_DIR = r"D:\kubota\Data\Model8\image_after"

OUT_DIR = r"D:\kubota\Data\Model10\alldata"

# T2欠損マスク（過去コード準拠）
T1_THRESH = 100.0
T2_ZERO_EPS = 0.0  # 厳密==0なら0.0でOK。ほぼ0判定にしたいなら 1e-6 等

# マスクは 0/1 を uint8（見やすくしたいなら255にする）
MASK_SCALE_255 = True

# ラベルは 0-14 をそのまま uint8
LABEL_PNG_DTYPE = np.uint8

# =========================


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def id_to_prefix(ixid: int) -> str:
    return f"IXI{ixid:03d}"


def load_nii(path: str) -> sitk.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return sitk.ReadImage(path)


def sitk_to_np(img: sitk.Image) -> np.ndarray:
    # (z, y, x)
    return sitk.GetArrayFromImage(img)


def make_t2_missing_mask_np(a1: np.ndarray, a2: np.ndarray,
                            t1_thresh: float = 100.0, t2_zero_eps: float = 0.0) -> np.ndarray:
    """
    欠損=1 を返す（T1は頭部っぽいのに、T2が0）
    a1,a2: (z,y,x)
    """
    if t2_zero_eps == 0.0:
        miss = (a1 > t1_thresh) & (a2 == 0)
    else:
        miss = (a1 > t1_thresh) & (a2 <= t2_zero_eps)
    return miss.astype(np.uint8)


def save_png_uint8(path: Path, arr2d_uint8: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), arr2d_uint8)
    if not ok:
        raise RuntimeError(f"Failed to write png: {path}")


def normalize_volume_to_uint8(vol_3d: np.ndarray) -> np.ndarray:
    """
    ボリュームを一括で 0-255 に線形正規化して uint8 にする。
    （スライスごとの正規化はしない）
    """
    x = vol_3d.astype(np.float32)
    lo = float(x.min())
    hi = float(x.max())
    if hi <= lo:
        return np.zeros_like(vol_3d, dtype=np.uint8)

    x = (x - lo) / (hi - lo)          # 0-1
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0 + 0.5).astype(np.uint8)
    return x


def export_one_case(ixid: int, age: float, out_root: Path) -> list[dict]:
    prefix = id_to_prefix(ixid)

    t1_path = os.path.join(IMAGE_DIR, f"{prefix}_T1_after.nii.gz")
    t2_path = os.path.join(IMAGE_DIR, f"{prefix}_T2_after.nii.gz")
    lab_path = os.path.join(LABEL_DIR, f"{prefix}_label_after.nii.gz")

    t1_img = load_nii(t1_path)
    t2_img = load_nii(t2_path)
    lab_img = load_nii(lab_path)

    a1 = sitk_to_np(t1_img)
    a2 = sitk_to_np(t2_img)
    al = sitk_to_np(lab_img)

    if a1.shape != a2.shape or a1.shape != al.shape:
        raise ValueError(
            f"Shape mismatch for {prefix}: "
            f"T1{a1.shape}, T2{a2.shape}, Label{al.shape}"
        )

    # 欠損マスク（3D, 生値で判定）
    t2_miss = make_t2_missing_mask_np(a1, a2, T1_THRESH, T2_ZERO_EPS)

    # 画像は「読み込んだ直後に」ボリューム一括で 0-255 に正規化してuint8化
    a1_u8 = normalize_volume_to_uint8(a1)
    a2_u8 = normalize_volume_to_uint8(a2)

    # 出力先：症例ごとにフォルダ
    case_dir = out_root / prefix
    img_dir = case_dir / "img"
    lab_dir = case_dir / "label"
    msk_dir = case_dir / "t2mask"
    ensure_dir(img_dir)
    ensure_dir(lab_dir)
    ensure_dir(msk_dir)

    records = []
    zdim = a1.shape[0]

    for z in range(zdim):
        # 2D slices（画像は既にuint8）
        t1_png = a1_u8[z]
        t2_png = a2_u8[z]

        lab_2d = al[z].astype(LABEL_PNG_DTYPE)
        lab_2d = (lab_2d * 18).astype(np.uint8)

        msk_2d = t2_miss[z].astype(np.uint8)
        if MASK_SCALE_255:
            msk_2d = (msk_2d * 255).astype(np.uint8)

        # file names
        t1_png_path = img_dir / f"{prefix}_z{z:03d}_T1.png"
        t2_png_path = img_dir / f"{prefix}_z{z:03d}_T2.png"
        lab_png_path = lab_dir / f"{prefix}_z{z:03d}_label.png"
        msk_png_path = msk_dir / f"{prefix}_z{z:03d}_T2mask.png"

        save_png_uint8(t1_png_path, t1_png)
        save_png_uint8(t2_png_path, t2_png)
        save_png_uint8(lab_png_path, lab_2d)
        save_png_uint8(msk_png_path, msk_2d)

        records.append({
            "IXI_ID": ixid,
            "age": age,
            "PREFIX": prefix,
            "slice": z,
            "t1_img": str(t1_png_path),
            "t2_img": str(t2_png_path),
            "label": str(lab_png_path),
            "t2mask": str(msk_png_path),
        })

    return records


def main():
    out_root = Path(OUT_DIR)
    ensure_dir(out_root)

    df = pd.read_csv(CSV_PATH)

    # まず症例ID列を「IXI_ID / ixi_id」優先で探す（idは最後）
    id_col = None
    for cand in ("IXI_ID", "ixi_id", "ID", "id"):
        if cand in df.columns:
            id_col = cand
            break
    if id_col is None:
        raise ValueError(f"ID column not found. Columns={df.columns.tolist()}")

    # age列を探す（大小区別しない）
    age_col = None
    for c in df.columns:
        if c.strip().lower() == "age":
            age_col = c
            break
    if age_col is None:
        raise ValueError(f"age column not found in CSV. Columns={df.columns.tolist()}")

    # 同一症例が複数行あっても壊れないように、症例ごとに最初のageを使う
    df_case = df[[id_col, age_col]].dropna().drop_duplicates(subset=[id_col])
    df_case[id_col] = df_case[id_col].astype(int)

    id_to_age = dict(zip(df_case[id_col], df_case[age_col]))
    ids = sorted(df_case[id_col].unique().tolist())

    all_records: list[dict] = []
    global_number = 0

    for ixid in ids:
        age = float(id_to_age[ixid])
        print(f"[export] IXI{ixid:03d}")

        rec = export_one_case(ixid, age, out_root)  # export_one_caseは age 引数版にしておく

        for r in rec:
            r["number"] = global_number
            global_number += 1

        all_records.extend(rec)

    out_csv = out_root / "paths_all_slices.csv"

    df_out = pd.DataFrame(all_records)

    # numberを一番左に固定
    cols = ["number"] + [c for c in df_out.columns if c != "number"]
    df_out = df_out[cols]

    df_out.to_csv(out_csv, index=False)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()