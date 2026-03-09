# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 10:20:35 2026

@author: kubota
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from typing import Optional
from scipy.stats import spearmanr

# ============================================================
# ✅ ここだけ編集すればOK（入出力・断面・表示レンジ）
# ============================================================
pid=304
CFG = {
    "cond_nii": f"D:/kubota/Data/Model10test/uniform_nii/{pid}_uniform.nii.gz",
    "label_nii": f"D:/kubota/Data/Model9/label14/IXI{pid:03d}_label_after.nii.gz",

    "slice_axial_z": None,
    "slice_coronal_y": None,
    "slice_sagittal_x": None,

    "cond_vmin": 0.0,
    "cond_vmax": 2.2,

    "label_min": 0,
    "label_max": 14,
    "label_cmap_name": "tab20",

    "boundary_widths": [1, 2],

    "out_csv": f"D:/Kubota/data/Model10test/cond_stats_{pid:03d}_uniform.csv",
    "out_png": None,

    "ignore_background_label0": False,
    
    "T1_nii": f"D:/kubota/Data/Model8/image_after/IXI{pid:03d}_T1_after.nii.gz",
    "T2_nii": f"D:/kubota/Data/Model8/image_after/IXI{pid:03d}_T2_after.nii.gz",
    
    "alpha": 0.3,
    
    "pix_vmin": 0.0,
    "pix_vmax": 1.0,
}
# ============================================================


def load_nii_img(path: str) -> nib.Nifti1Image:
    return nib.load(path)


def make_label_cmap(n_labels: int, cmap_name: str = "tab20"):
    base = plt.get_cmap(cmap_name)
    colors = [base(i % base.N) for i in range(n_labels)]
    return plt.matplotlib.colors.ListedColormap(colors, name=f"labels_{n_labels}")


def rotate_cond_to_match_label(cond_zyx: np.ndarray) -> np.ndarray:
    """
    cond (Z,Y,X) を label (Z,Y,X) に合わせるための固定操作。

    今回の mismatch:
      cond_rot: (256,251,256)
      label   : (256,256,251)

    → Y と X が入れ替わっているので (1,2) を transpose する。
    """
    v = cond_zyx

    # ★ まず Y と X を入れ替える（Zはそのまま）
    v = np.transpose(v, (1, 0, 2))

    # 必要ならこの後に回転/反転を追加（まずは shape を一致させるのが先）
    # v = np.rot90(v, k=1, axes=(1, 2))
    # v = np.flip(v, axis=2)

    return v


def show_slices_3x3(cond_zyx: np.ndarray, lab_zyx: np.ndarray, pix_zyx: np.ndarray,
                    z: int, y: int, x: int,
                    cond_vmin: float, cond_vmax: float,
                    pix_vmin: float, pix_vmax: float,
                    label_min: int, label_max: int,
                    cmap_name: str,
                    out_png: Optional[str]):

    Z, Y, X = cond_zyx.shape
    z = int(np.clip(z, 0, Z - 1))
    y = int(np.clip(y, 0, Y - 1))
    x = int(np.clip(x, 0, X - 1))

    # axial
    cond_ax = cond_zyx[z, :, :]
    lab_ax  = lab_zyx[z, :, :]
    pix_ax  = pix_zyx[z, :, :]

    # coronal
    cond_cor = cond_zyx[:, y, :]
    lab_cor  = lab_zyx[:, y, :]
    pix_cor  = pix_zyx[:, y, :]

    # sagittal
    cond_sag = cond_zyx[:, :, x]
    lab_sag  = lab_zyx[:, :, x]
    pix_sag  = pix_zyx[:, :, x]

    # label colormap (0..14)
    n_labels = (label_max - label_min + 1)
    lab_cmap = make_label_cmap(n_labels, cmap_name=cmap_name)
    boundaries = np.arange(label_min - 0.5, label_max + 1.5, 1.0)
    lab_norm = plt.matplotlib.colors.BoundaryNorm(boundaries=boundaries, ncolors=n_labels)

    # ------------------------------------------------------------
    # ★ここがポイント：
    # 3列(画像) + 1列(カラーバー専用) の Grid にして、
    # 画像の表示領域を必ず揃える（labelだけズレる問題が消える）
    # ------------------------------------------------------------
    fig, ax = plt.subplots(
        3, 4,
        figsize=(10, 10),  # 横幅を詰める
        gridspec_kw={"width_ratios": [1, 1, 1, 0.05], "wspace": 0.06, "hspace": 0.12}
    )

    # 画像表示用 axes（0..2列）
    a00, a01, a02 = ax[0, 0], ax[0, 1], ax[0, 2]
    a10, a11, a12 = ax[1, 0], ax[1, 1], ax[1, 2]
    a20, a21, a22 = ax[2, 0], ax[2, 1], ax[2, 2]

    # カラーバー用 axes（3列）
    c0, c1, c2 = ax[0, 3], ax[1, 3], ax[2, 3]

    # ----------------
    # Row 0: Label（カラーバー列は消す）
    # ----------------
    a00.imshow(lab_ax, cmap=lab_cmap, norm=lab_norm, interpolation="nearest", aspect="equal")
    a00.set_title(f"Label (Axial z={z})"); a00.axis("off")

    a01.imshow(lab_cor, cmap=lab_cmap, norm=lab_norm, interpolation="nearest", aspect="equal")
    a01.set_title(f"Label (Coronal y={y})"); a01.axis("off")

    a02.imshow(lab_sag, cmap=lab_cmap, norm=lab_norm, interpolation="nearest", aspect="equal")
    a02.set_title(f"Label (Sagittal x={x})"); a02.axis("off")

    c0.axis("off")  # label行のカラーバー領域は空にする（幅は維持）

    # ----------------
    # Row 1: Conductivity（行で共通のカラーバー1本）
    # ----------------
    im10 = a10.imshow(cond_ax, cmap="jet", vmin=cond_vmin, vmax=cond_vmax, interpolation="nearest", aspect="equal")
    a10.set_title(f"Cond (Axial z={z})"); a10.axis("off")

    a11.imshow(cond_cor, cmap="jet", vmin=cond_vmin, vmax=cond_vmax, interpolation="nearest", aspect="equal")
    a11.set_title(f"Cond (Coronal y={y})"); a11.axis("off")

    a12.imshow(cond_sag, cmap="jet", vmin=cond_vmin, vmax=cond_vmax, interpolation="nearest", aspect="equal")
    a12.set_title(f"Cond (Sagittal x={x})"); a12.axis("off")

    fig.colorbar(im10, cax=c1)

    # ----------------
    # Row 2: PixelMap（行で共通のカラーバー1本）
    # ----------------
    im20 = a20.imshow(pix_ax, cmap="gray", vmin=pix_vmin, vmax=pix_vmax, interpolation="nearest", aspect="equal")
    a20.set_title(f"PixelMap (Axial z={z})"); a20.axis("off")

    a21.imshow(pix_cor, cmap="gray", vmin=pix_vmin, vmax=pix_vmax, interpolation="nearest", aspect="equal")
    a21.set_title(f"PixelMap (Coronal y={y})"); a21.axis("off")

    a22.imshow(pix_sag, cmap="gray", vmin=pix_vmin, vmax=pix_vmax, interpolation="nearest", aspect="equal")
    a22.set_title(f"PixelMap (Sagittal x={x})"); a22.axis("off")

    fig.colorbar(im20, cax=c2)

    # 余白をさらに削る（tight_layoutより効く）
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.03)

    if out_png:
        out_dir = os.path.dirname(out_png)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_png, dpi=200, bbox_inches="tight")

    plt.show()


def plot_scatter_triplet_by_label(
    cond: np.ndarray,
    pix: np.ndarray,
    lab: np.ndarray,
    core1_mask: np.ndarray,
    core2_mask: np.ndarray,
    label_min: int = 0,
    label_max: int = 14,
    n_per_label: int = 2000,
    ignore_label0: bool = False,
    title_prefix: str = "",
    cmap_name: str = "tab20",
):
    """
    3つの散布図（all / core1 / core2）を横に並べて表示。
    14部位(0..14)を色分けし、図の外に共通凡例を出す。

    cond, pix, lab は同shapeの3D配列を想定。
    core1_mask, core2_mask は lab と同shapeのbool配列。
    """

    assert cond.shape == pix.shape == lab.shape, "cond/pix/lab shape mismatch"
    assert core1_mask.shape == lab.shape and core2_mask.shape == lab.shape, "core mask shape mismatch"

    # region masks
    m_all  = np.ones(lab.shape, dtype=bool)
    m_c1   = core1_mask.astype(bool)
    m_c2   = core2_mask.astype(bool)

    # labels
    labels = list(range(label_min, label_max + 1))
    if ignore_label0 and 0 in labels:
        labels.remove(0)

    # color map fixed（ラベル表示と同じ色割り当てにする）
    cmap = plt.get_cmap(CFG["label_cmap_name"])
    colors = cmap.colors
    color_of = {lid: colors[lid] for lid in range(label_min, label_max + 1)}

    # plot setup
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True, sharey=True)
    regions = [("all", m_all), ("core1", m_c1), ("core2", m_c2)]

    # legend handles (共通凡例用)
    handles = []

    for ax, (rname, rmask) in zip(axes, regions):
        for lid in labels:
            m = (lab == lid) & rmask
            n = int(np.count_nonzero(m))
            if n == 0:
                continue

            x = pix[m]
            y = cond[m]

            # サンプリング（重すぎ回避）
            if n > n_per_label:
                idx = np.random.choice(n, n_per_label, replace=False)
                x = x[idx]
                y = y[idx]

            ax.scatter(
                x, y,
                s=4,
                alpha=0.25,
                color=color_of[lid],
                linewidths=0
            )

        ax.set_title(f"{title_prefix}{rname}")
        ax.set_xlabel("PixelMap")
        ax.grid(True, linewidth=0.5, alpha=0.4)

    axes[0].set_ylabel("Conductivity")

    # ---- 共通凡例（色がどれかを必ず表示）----
    # 1回だけ作る（存在するラベルだけ）
    for lid in labels:
        # どのregionでも出てこないラベルは凡例に入れない
        exists = np.any(lab == lid)
        if not exists:
            continue
        h = plt.Line2D([0], [0], marker='o', linestyle='',
                       markersize=6, color=color_of[lid], label=f"Label {lid}")
        handles.append(h)

    # 図の外側に凡例を配置（右側）
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        frameon=False,
        ncol=5
    )
    
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    plt.show()

    # 色対応をコンソールにも出したい場合
    #print("Label -> Color (RGBA)")
    #for lid in labels:
    #    print(lid, color_of[lid])
        
def core_mask(label_zyx: np.ndarray, label_id: int, erode_iter: int) -> np.ndarray:
    m = (label_zyx == label_id)
    if erode_iter <= 0:
        return m
    structure = np.ones((3, 3, 3), dtype=bool)  # 26近傍
    return binary_erosion(m, structure=structure, iterations=erode_iter, border_value=0)


def stats_for_mask(cond: np.ndarray, mask: np.ndarray):
    vals = cond[mask]
    if vals.size == 0:
        return np.nan, np.nan, 0
    return float(vals.mean()), float(vals.std(ddof=0)), int(vals.size)

def create_pixel_map(t1: np.ndarray, t2: np.ndarray, alpha: float) -> np.ndarray:
    """
    T1, T2 を最大値で正規化し
    (1-α)*(1-T1) + α*T2 の PixelMap を作成
    """
    t1n = t1 / np.max(t1) if np.max(t1) != 0 else t1
    t2n = t2 / np.max(t2) if np.max(t2) != 0 else t2

    return (1 - alpha) * (1 - t1n) + alpha * t2n

def spearman_for_mask(cond: np.ndarray, pix: np.ndarray, mask: np.ndarray):
    v1 = cond[mask]
    v2 = pix[mask]

    if v1.size < 2:
        return np.nan, np.nan

    if np.all(v1 == v1[0]) or np.all(v2 == v2[0]):
        return np.nan, np.nan

    r, p = spearmanr(v1, v2)
    return float(r), float(p)

def main():
    # ① 読み込み
    cond_img = load_nii_img(CFG["cond_nii"])
    label_img = load_nii_img(CFG["label_nii"])
    t1_img = load_nii_img(CFG["T1_nii"])
    t2_img = load_nii_img(CFG["T2_nii"])

    cond = cond_img.get_fdata().astype(np.float32)
    lab = label_img.get_fdata()
    t1 = t1_img.get_fdata().astype(np.float32)
    t2 = t2_img.get_fdata().astype(np.float32)

    # ラベルは整数化（微小誤差対策）
    lab_int = np.rint(lab).astype(np.int32)

    # ★回転はしない：あなたが確定させた軸入れ替えだけ適用（x,y,z -> y,x,z）
    # cond/t1/t2 を label と同じ軸順に揃える
    cond_aligned = np.transpose(cond, (1, 0, 2))

    # shapeチェック
    if cond_aligned.shape != lab_int.shape:
        raise ValueError(f"Shape mismatch: cond{cond_aligned.shape} vs label{lab_int.shape}")
    if t1.shape != lab_int.shape:
        raise ValueError(f"Shape mismatch: T1{t1.shape} vs label{lab_int.shape}")
    if t2.shape != lab_int.shape:
        raise ValueError(f"Shape mismatch: T2{t2.shape} vs label{lab_int.shape}")

    # ② PixelMap 作成
    alpha = float(CFG["alpha"])
    pix_map = create_pixel_map(t1, t2, alpha=alpha)

    # 断面 index
    Z, Y, X = cond_aligned.shape
    z = CFG["slice_axial_z"] if CFG["slice_axial_z"] is not None else (Z // 2)
    y = CFG["slice_coronal_y"] if CFG["slice_coronal_y"] is not None else (Y // 2)
    x = CFG["slice_sagittal_x"] if CFG["slice_sagittal_x"] is not None else (X // 2)

    # ② 表示（3×3：Label / Cond / PixelMap）
    show_slices_3x3(
        cond_zyx=cond_aligned,
        lab_zyx=lab_int,
        pix_zyx=pix_map,
        z=z, y=y, x=x,
        cond_vmin=float(CFG["cond_vmin"]),
        cond_vmax=float(CFG["cond_vmax"]),
        pix_vmin=float(CFG.get("pix_vmin", 0.0)),
        pix_vmax=float(CFG.get("pix_vmax", 1.0)),
        label_min=int(CFG["label_min"]),
        label_max=int(CFG["label_max"]),
        cmap_name=str(CFG["label_cmap_name"]),
        out_png=CFG["out_png"]
    )
    
    # core1/core2 マスクを作る（全ラベルまとめて）
    m_core1 = np.zeros_like(lab_int, dtype=bool)
    m_core2 = np.zeros_like(lab_int, dtype=bool)
    
    for lid in range(int(CFG["label_min"]), int(CFG["label_max"]) + 1):
        if CFG["ignore_background_label0"] and lid == 0:
            continue
        m_core1 |= core_mask(lab_int, lid, erode_iter=1)
        m_core2 |= core_mask(lab_int, lid, erode_iter=2)
    
    plot_scatter_triplet_by_label(
        cond=cond_aligned,          # or cond_aligned など
        pix=pix_map,
        lab=lab_int,
        core1_mask=m_core1,
        core2_mask=m_core2,
        label_min=CFG["label_min"],
        label_max=CFG["label_max"],
        n_per_label=2000,
        ignore_label0=CFG["ignore_background_label0"],
        title_prefix=f"PID {pid} - "
    )

    # ③④ 統計（all, core1, core2, boundary1, boundary2）＋ Spearman(cond vs pix)
    label_min = int(CFG["label_min"])
    label_max = int(CFG["label_max"])
    widths = list(CFG["boundary_widths"])

    labels = list(range(label_min, label_max + 1))
    if CFG["ignore_background_label0"] and 0 in labels:
        labels.remove(0)

    def spearman_for_mask(a: np.ndarray, b: np.ndarray, m: np.ndarray):
        v1 = a[m]
        v2 = b[m]
        if v1.size < 2:
            return np.nan, np.nan
        if np.all(v1 == v1[0]) or np.all(v2 == v2[0]):
            return np.nan, np.nan
        r, p = spearmanr(v1, v2)
        return float(r), float(p)

    rows = []
    for label_id in labels:
        m_all = (lab_int == label_id)

        cores = {}
        boundaries = {}
        for w in widths:
            m_core = core_mask(lab_int, label_id, erode_iter=w)
            m_bnd = m_all & (~m_core)
            cores[w] = m_core
            boundaries[w] = m_bnd

        # all
        mean_all, std_all, n_all = stats_for_mask(cond_aligned, m_all)
        pix_mean_all, pix_std_all, _ = stats_for_mask(pix_map, m_all)
        r_all, p_all = spearman_for_mask(cond_aligned, pix_map, m_all)
        rows.append({
            "label": label_id, "region": "all",
            "cond_mean": mean_all, "cond_std": std_all,
            "pix_mean": pix_mean_all, "pix_std": pix_std_all,
            "spearman_r": r_all, "spearman_p": p_all,
            "n_vox": n_all
        })

        # cores
        for w in widths:
            mean_c, std_c, n_c = stats_for_mask(cond_aligned, cores[w])
            pix_mean_c, pix_std_c, _ = stats_for_mask(pix_map, cores[w])
            r_c, p_c = spearman_for_mask(cond_aligned, pix_map, cores[w])
            rows.append({
                "label": label_id, "region": f"core{w}",
                "cond_mean": mean_c, "cond_std": std_c,
                "pix_mean": pix_mean_c, "pix_std": pix_std_c,
                "spearman_r": r_c, "spearman_p": p_c,
                "n_vox": n_c
            })

        # boundaries
        for w in widths:
            mean_b, std_b, n_b = stats_for_mask(cond_aligned, boundaries[w])
            pix_mean_b, pix_std_b, _ = stats_for_mask(pix_map, boundaries[w])
            r_b, p_b = spearman_for_mask(cond_aligned, pix_map, boundaries[w])
            rows.append({
                "label": label_id, "region": f"boundary{w}",
                "cond_mean": mean_b, "cond_std": std_b,
                "pix_mean": pix_mean_b, "pix_std": pix_std_b,
                "spearman_r": r_b, "spearman_p": p_b,
                "n_vox": n_b
            })

    df = pd.DataFrame(rows)

    out_csv = CFG["out_csv"]
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_csv, index=False, float_format="%.8f")

    print(f"Saved CSV: {out_csv}")
    print(df.head(15))


if __name__ == "__main__":
    main()