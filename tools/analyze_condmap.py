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
from scipy.stats import spearmanr
from typing import Optional


# ============================================================
# ✅ ここだけ編集すればOK
# ============================================================
CFG = {
    # -----------------------------
    # 処理したい症例IDを指定
    # -----------------------------
    "pids": [46, 122, 254, 304, 362, 369, 401, 437, 441, 575],

    # -----------------------------
    # 入力パス（pidを埋め込む）
    # cond は {pid} → 46 のようにそのまま
    # label/T1/T2 は {pid:03d} → 046 のように3桁
    # -----------------------------
    "cond_nii_tmpl":  "D:/kubota/Data/Model10test/M3_099_02_nii/{pid}_pred.nii.gz",
    "label_nii_tmpl": "D:/kubota/Data/Model9/label14/IXI{pid:03d}_label_after.nii.gz",
    "T1_nii_tmpl":    "D:/kubota/Data/Model8/image_after/IXI{pid:03d}_T1_after.nii.gz",
    "T2_nii_tmpl":    "D:/kubota/Data/Model8/image_after/IXI{pid:03d}_T2_after.nii.gz",

    # -----------------------------
    # 出力先
    # -----------------------------
    "out_csv_dir": "D:/Kubota/data/Model10test/csv/M3_099_02",
    "out_excel": "D:/Kubota/data/Model10test/csv/M3_099_02/label_summary.xlsx",

    # -----------------------------
    # 表示・解析設定
    # -----------------------------
    "slice_axial_z": None,
    "slice_coronal_y": None,
    "slice_sagittal_x": None,

    "cond_vmin": 0.0,
    "cond_vmax": 2.2,

    "pix_vmin": 0.0,
    "pix_vmax": 1.0,

    "label_min": 0,
    "label_max": 14,
    "label_cmap_name": "tab20",

    # ここでは表を All / Core1 / Core2 にする
    "excel_target_labels": list(range(1, 15)),   # 1〜14 をシート出力
    "ignore_background_label0": False,

    "alpha": 0.3,

    # 図を出すか
    "show_figures": True,
    "save_case_png": False,
    "out_png_dir": "D:/Kubota/data/Model10test/png",
}
# ============================================================


def load_nii_img(path: str) -> nib.Nifti1Image:
    return nib.load(path)


def make_label_cmap(n_labels: int, cmap_name: str = "tab20"):
    base = plt.get_cmap(cmap_name)
    colors = [base(i % base.N) for i in range(n_labels)]
    return plt.matplotlib.colors.ListedColormap(colors, name=f"labels_{n_labels}")


def show_slices_3x3(
    cond_zyx: np.ndarray,
    lab_zyx: np.ndarray,
    pix_zyx: np.ndarray,
    z: int,
    y: int,
    x: int,
    cond_vmin: float,
    cond_vmax: float,
    pix_vmin: float,
    pix_vmax: float,
    label_min: int,
    label_max: int,
    cmap_name: str,
    out_png: Optional[str]
):
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

    n_labels = (label_max - label_min + 1)
    lab_cmap = make_label_cmap(n_labels, cmap_name=cmap_name)
    boundaries = np.arange(label_min - 0.5, label_max + 1.5, 1.0)
    lab_norm = plt.matplotlib.colors.BoundaryNorm(boundaries=boundaries, ncolors=n_labels)

    fig, ax = plt.subplots(
        3, 4,
        figsize=(10, 10),
        gridspec_kw={"width_ratios": [1, 1, 1, 0.05], "wspace": 0.06, "hspace": 0.12}
    )

    a00, a01, a02 = ax[0, 0], ax[0, 1], ax[0, 2]
    a10, a11, a12 = ax[1, 0], ax[1, 1], ax[1, 2]
    a20, a21, a22 = ax[2, 0], ax[2, 1], ax[2, 2]

    c0, c1, c2 = ax[0, 3], ax[1, 3], ax[2, 3]

    a00.imshow(lab_ax, cmap=lab_cmap, norm=lab_norm, interpolation="nearest", aspect="equal")
    a00.set_title(f"Label (Axial z={z})"); a00.axis("off")

    a01.imshow(lab_cor, cmap=lab_cmap, norm=lab_norm, interpolation="nearest", aspect="equal")
    a01.set_title(f"Label (Coronal y={y})"); a01.axis("off")

    a02.imshow(lab_sag, cmap=lab_cmap, norm=lab_norm, interpolation="nearest", aspect="equal")
    a02.set_title(f"Label (Sagittal x={x})"); a02.axis("off")

    c0.axis("off")

    im10 = a10.imshow(cond_ax, cmap="jet", vmin=cond_vmin, vmax=cond_vmax, interpolation="nearest", aspect="equal")
    a10.set_title(f"Cond (Axial z={z})"); a10.axis("off")

    a11.imshow(cond_cor, cmap="jet", vmin=cond_vmin, vmax=cond_vmax, interpolation="nearest", aspect="equal")
    a11.set_title(f"Cond (Coronal y={y})"); a11.axis("off")

    a12.imshow(cond_sag, cmap="jet", vmin=cond_vmin, vmax=cond_vmax, interpolation="nearest", aspect="equal")
    a12.set_title(f"Cond (Sagittal x={x})"); a12.axis("off")

    fig.colorbar(im10, cax=c1)

    im20 = a20.imshow(pix_ax, cmap="gray", vmin=pix_vmin, vmax=pix_vmax, interpolation="nearest", aspect="equal")
    a20.set_title(f"PixelMap (Axial z={z})"); a20.axis("off")

    a21.imshow(pix_cor, cmap="gray", vmin=pix_vmin, vmax=pix_vmax, interpolation="nearest", aspect="equal")
    a21.set_title(f"PixelMap (Coronal y={y})"); a21.axis("off")

    a22.imshow(pix_sag, cmap="gray", vmin=pix_vmin, vmax=pix_vmax, interpolation="nearest", aspect="equal")
    a22.set_title(f"PixelMap (Sagittal x={x})"); a22.axis("off")

    fig.colorbar(im20, cax=c2)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.03)

    if out_png:
        out_dir = os.path.dirname(out_png)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_png, dpi=200, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def core_mask(label_zyx: np.ndarray, label_id: int, erode_iter: int) -> np.ndarray:
    m = (label_zyx == label_id)
    if erode_iter <= 0:
        return m
    structure = np.ones((3, 3, 3), dtype=bool)
    return binary_erosion(m, structure=structure, iterations=erode_iter, border_value=0)


def stats_for_mask(cond: np.ndarray, mask: np.ndarray):
    vals = cond[mask]
    if vals.size == 0:
        return np.nan, np.nan, 0
    return float(vals.mean()), float(vals.std(ddof=0)), int(vals.size)


def spearman_for_mask(cond: np.ndarray, pix: np.ndarray, mask: np.ndarray):
    v1 = cond[mask]
    v2 = pix[mask]

    if v1.size < 2:
        return np.nan, np.nan
    if np.all(v1 == v1[0]) or np.all(v2 == v2[0]):
        return np.nan, np.nan

    r, p = spearmanr(v1, v2)
    return float(r), float(p)


def create_pixel_map(t1: np.ndarray, t2: np.ndarray, alpha: float) -> np.ndarray:
    t1n = t1 / np.max(t1) if np.max(t1) != 0 else t1
    t2n = t2 / np.max(t2) if np.max(t2) != 0 else t2
    return (1 - alpha) * (1 - t1n) + alpha * t2n


def format_path(template: str, pid: int) -> str:
    return template.format(pid=pid)


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
    ラベルごとに色分けし、共通凡例を下に表示。
    """

    assert cond.shape == pix.shape == lab.shape, "cond/pix/lab shape mismatch"
    assert core1_mask.shape == lab.shape and core2_mask.shape == lab.shape, "core mask shape mismatch"

    # region masks
    m_all = np.ones(lab.shape, dtype=bool)
    m_c1 = core1_mask.astype(bool)
    m_c2 = core2_mask.astype(bool)

    labels = list(range(label_min, label_max + 1))
    if ignore_label0 and 0 in labels:
        labels.remove(0)

    # 色設定
    cmap = plt.get_cmap(cmap_name)
    if hasattr(cmap, "colors"):
        colors = cmap.colors
        color_of = {lid: colors[lid % len(colors)] for lid in range(label_min, label_max + 1)}
    else:
        color_of = {lid: cmap((lid - label_min) / max(1, (label_max - label_min))) for lid in range(label_min, label_max + 1)}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True, sharey=True)
    regions = [("all", m_all), ("core1", m_c1), ("core2", m_c2)]

    handles = []

    for ax, (rname, rmask) in zip(axes, regions):
        for lid in labels:
            m = (lab == lid) & rmask
            n = int(np.count_nonzero(m))
            if n == 0:
                continue

            x = pix[m]
            y = cond[m]

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

    for lid in labels:
        exists = np.any(lab == lid)
        if not exists:
            continue
        h = plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markersize=6,
            color=color_of[lid],
            label=f"Label {lid}"
        )
        handles.append(h)

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
    plt.close(fig)

def process_one_case(pid: int, cfg: dict) -> pd.DataFrame:
    print(f"\n===== Processing PID {pid} =====")

    cond_path  = format_path(cfg["cond_nii_tmpl"], pid)
    label_path = format_path(cfg["label_nii_tmpl"], pid)
    t1_path    = format_path(cfg["T1_nii_tmpl"], pid)
    t2_path    = format_path(cfg["T2_nii_tmpl"], pid)

    for p in [cond_path, label_path, t1_path, t2_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")

    cond_img = load_nii_img(cond_path)
    label_img = load_nii_img(label_path)
    t1_img = load_nii_img(t1_path)
    t2_img = load_nii_img(t2_path)

    cond = cond_img.get_fdata().astype(np.float32)
    lab = label_img.get_fdata()
    t1 = t1_img.get_fdata().astype(np.float32)
    t2 = t2_img.get_fdata().astype(np.float32)

    lab_int = np.rint(lab).astype(np.int32)

    # cond を label に合わせる
    cond_aligned = np.transpose(cond, (1, 0, 2))

    if cond_aligned.shape != lab_int.shape:
        raise ValueError(f"Shape mismatch: cond{cond_aligned.shape} vs label{lab_int.shape}")
    if t1.shape != lab_int.shape:
        raise ValueError(f"Shape mismatch: T1{t1.shape} vs label{lab_int.shape}")
    if t2.shape != lab_int.shape:
        raise ValueError(f"Shape mismatch: T2{t2.shape} vs label{lab_int.shape}")

    pix_map = create_pixel_map(t1, t2, alpha=float(cfg["alpha"]))

    # --------------------------------------------------
    # 図表示用 index
    # bool が入る事故を避ける
    # --------------------------------------------------
    Z, Y, X = cond_aligned.shape

    def _resolve_slice_index(v, size):
        if v is None:
            return size // 2
        if isinstance(v, bool):
            return size // 2
        return int(np.clip(v, 0, size - 1))

    z = _resolve_slice_index(cfg.get("slice_axial_z", None), Z)
    y = _resolve_slice_index(cfg.get("slice_coronal_y", None), Y)
    x = _resolve_slice_index(cfg.get("slice_sagittal_x", None), X)

    # --------------------------------------------------
    # 図表示
    # 断面図 + 前から使っていた scatter 図を両方出す
    # --------------------------------------------------
    if cfg.get("show_figures", False):
        out_png = None
        if cfg.get("save_case_png", False):
            out_png = os.path.join(cfg["out_png_dir"], f"case_{pid}_overview.png")

        show_slices_3x3(
            cond_zyx=cond_aligned,
            lab_zyx=lab_int,
            pix_zyx=pix_map,
            z=z, y=y, x=x,
            cond_vmin=float(cfg["cond_vmin"]),
            cond_vmax=float(cfg["cond_vmax"]),
            pix_vmin=float(cfg["pix_vmin"]),
            pix_vmax=float(cfg["pix_vmax"]),
            label_min=int(cfg["label_min"]),
            label_max=int(cfg["label_max"]),
            cmap_name=str(cfg["label_cmap_name"]),
            out_png=out_png
        )

        # ---- 前から使っていた scatter 用の core mask（全ラベルまとめて）----
        m_core1 = np.zeros_like(lab_int, dtype=bool)
        m_core2 = np.zeros_like(lab_int, dtype=bool)

        for lid in range(int(cfg["label_min"]), int(cfg["label_max"]) + 1):
            if cfg["ignore_background_label0"] and lid == 0:
                continue
            m_core1 |= core_mask(lab_int, lid, erode_iter=1)
            m_core2 |= core_mask(lab_int, lid, erode_iter=2)

        plot_scatter_triplet_by_label(
            cond=cond_aligned,
            pix=pix_map,
            lab=lab_int,
            core1_mask=m_core1,
            core2_mask=m_core2,
            label_min=int(cfg["label_min"]),
            label_max=int(cfg["label_max"]),
            n_per_label=2000,
            ignore_label0=bool(cfg["ignore_background_label0"]),
            title_prefix=f"PID {pid} - ",
            cmap_name=str(cfg["label_cmap_name"]),
        )

    # --------------------------------------------------
    # 統計
    # --------------------------------------------------
    label_min = int(cfg["label_min"])
    label_max = int(cfg["label_max"])
    labels = list(range(label_min, label_max + 1))
    if cfg["ignore_background_label0"] and 0 in labels:
        labels.remove(0)

    rows = []
    for label_id in labels:
        m_all = (lab_int == label_id)

        # All
        cond_mean_all, cond_std_all, n_all = stats_for_mask(cond_aligned, m_all)
        pix_mean_all, pix_std_all, _ = stats_for_mask(pix_map, m_all)
        r_all, p_all = spearman_for_mask(cond_aligned, pix_map, m_all)

        rows.append({
            "pid": pid,
            "label": label_id,
            "region": "all",
            "cond_mean": cond_mean_all,
            "cond_std": cond_std_all,
            "pix_mean": pix_mean_all,
            "pix_std": pix_std_all,
            "spearman_r": r_all,
            "spearman_p": p_all,
            "n_vox": n_all
        })

        # Core1
        m_core1 = core_mask(lab_int, label_id, erode_iter=1)
        cond_mean_c1, cond_std_c1, n_c1 = stats_for_mask(cond_aligned, m_core1)
        pix_mean_c1, pix_std_c1, _ = stats_for_mask(pix_map, m_core1)
        r_c1, p_c1 = spearman_for_mask(cond_aligned, pix_map, m_core1)

        rows.append({
            "pid": pid,
            "label": label_id,
            "region": "core1",
            "cond_mean": cond_mean_c1,
            "cond_std": cond_std_c1,
            "pix_mean": pix_mean_c1,
            "pix_std": pix_std_c1,
            "spearman_r": r_c1,
            "spearman_p": p_c1,
            "n_vox": n_c1
        })

        # Core2
        m_core2 = core_mask(lab_int, label_id, erode_iter=2)
        cond_mean_c2, cond_std_c2, n_c2 = stats_for_mask(cond_aligned, m_core2)
        pix_mean_c2, pix_std_c2, _ = stats_for_mask(pix_map, m_core2)
        r_c2, p_c2 = spearman_for_mask(cond_aligned, pix_map, m_core2)

        rows.append({
            "pid": pid,
            "label": label_id,
            "region": "core2",
            "cond_mean": cond_mean_c2,
            "cond_std": cond_std_c2,
            "pix_mean": pix_mean_c2,
            "pix_std": pix_std_c2,
            "spearman_r": r_c2,
            "spearman_p": p_c2,
            "n_vox": n_c2
        })

    df_case = pd.DataFrame(rows)

    out_csv_dir = cfg["out_csv_dir"]
    os.makedirs(out_csv_dir, exist_ok=True)
    out_csv = os.path.join(out_csv_dir, f"cond_stats_{pid:03d}.csv")
    df_case.to_csv(out_csv, index=False, float_format="%.8f")

    print(f"Saved case CSV: {out_csv}")
    return df_case



def build_label_sheet_table(df_all: pd.DataFrame, label_id: int, regions=("all", "core1", "core2")):
    """
    1ラベル分のシート用データを作る
    戻り値:
      main_rows: Mean/SD/CV 表
      sp_rows  : Spearman表
    """
    dfl = df_all[df_all["label"] == label_id].copy()
    dfl["cond_cv"] = dfl["cond_std"] / dfl["cond_mean"]

    subjects = sorted(dfl["pid"].dropna().unique().tolist())

    region_title = {"all": "All", "core1": "Core1", "core2": "Core2"}

    # ---- 上段: Mean / SD / CV ----
    main_rows = []
    header1 = ["Subject"]
    header2 = [""]

    for r in regions:
        rt = region_title[r]
        header1 += [rt, "", ""]
        header2 += ["Mean", "SD", "CV"]

    main_rows.append(header1)
    main_rows.append(header2)

    for pid in subjects:
        row = [pid]
        dfi = dfl[dfl["pid"] == pid]

        for r in regions:
            tmp = dfi[dfi["region"] == r]
            if len(tmp) == 0:
                row += [np.nan, np.nan, np.nan]
            else:
                mean_v = float(tmp["cond_mean"].iloc[0]) if pd.notna(tmp["cond_mean"].iloc[0]) else np.nan
                std_v  = float(tmp["cond_std"].iloc[0]) if pd.notna(tmp["cond_std"].iloc[0]) else np.nan
                cv_v   = float(tmp["cond_cv"].iloc[0]) if pd.notna(tmp["cond_cv"].iloc[0]) else np.nan
                row += [mean_v, std_v, cv_v]

        main_rows.append(row)

    # Average行
    avg_row = ["Average"]
    for r in regions:
        vals_mean, vals_std, vals_cv = [], [], []
        col_offset = 1 + 3 * list(regions).index(r)

        for rr in main_rows[2:]:
            vals_mean.append(rr[col_offset + 0])
            vals_std.append(rr[col_offset + 1])
            vals_cv.append(rr[col_offset + 2])

        avg_row += [
            np.nanmean(vals_mean) if len(vals_mean) else np.nan,
            np.nanmean(vals_std) if len(vals_std) else np.nan,
            np.nanmean(vals_cv) if len(vals_cv) else np.nan,
        ]

    main_rows.append(avg_row)

    # ---- 下段: Spearman ----
    sp_rows = []
    sp_header = ["Subject"] + [region_title[r] for r in regions]
    sp_rows.append(sp_header)

    for pid in subjects:
        row = [pid]
        dfi = dfl[dfl["pid"] == pid]

        for r in regions:
            tmp = dfi[dfi["region"] == r]
            if len(tmp) == 0:
                row.append(np.nan)
            else:
                row.append(float(tmp["spearman_r"].iloc[0]) if pd.notna(tmp["spearman_r"].iloc[0]) else np.nan)
        sp_rows.append(row)

    avg_sp = ["Average"]
    for j in range(1, len(sp_header)):
        vals = [rr[j] for rr in sp_rows[1:]]
        avg_sp.append(np.nanmean(vals) if len(vals) else np.nan)
    sp_rows.append(avg_sp)

    return main_rows, sp_rows


def write_label_summary_excel(
    df_all: pd.DataFrame,
    out_excel: str,
    label_ids,
    regions=("all", "core1", "core2")
):
    """
    ラベルごとにシートを分けてExcel出力
    """
    out_dir = os.path.dirname(out_excel)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        for label_id in label_ids:
            main_rows, sp_rows = build_label_sheet_table(df_all, label_id, regions=regions)

            sheet_name = f"Label_{label_id}"

            # まず main_rows を DataFrame 化して出力
            df_main_sheet = pd.DataFrame(main_rows)
            df_main_sheet.to_excel(writer, sheet_name=sheet_name, header=False, index=False, startrow=0)

            # Spearman見出し
            startrow_sp_title = len(main_rows) + 2
            startrow_sp_table = startrow_sp_title + 1

            ws = writer.sheets[sheet_name]
            ws.cell(row=startrow_sp_title + 1, column=1, value="スピアマン相関係数")

            # Spearman表
            df_sp_sheet = pd.DataFrame(sp_rows)
            df_sp_sheet.to_excel(writer, sheet_name=sheet_name, header=False, index=False, startrow=startrow_sp_table)

        # 縦長の全データ
        df_all.to_excel(writer, sheet_name="All_Cases_Long", index=False)

        # summary
        df_summary = (
            df_all
            .groupby(["label", "region"], dropna=False)
            .agg(
                cond_mean_mean=("cond_mean", "mean"),
                cond_std_mean=("cond_std", "mean"),
                spearman_r_mean=("spearman_r", "mean"),
                n_vox_mean=("n_vox", "mean"),
                n_cases=("pid", "nunique")
            )
            .reset_index()
        )
        df_summary["cond_cv_mean"] = df_summary["cond_std_mean"] / df_summary["cond_mean_mean"]
        df_summary.to_excel(writer, sheet_name="Summary_Long", index=False)

    print(f"Saved Excel: {out_excel}")


def main():
    all_case_dfs = []

    for pid in CFG["pids"]:
        try:
            df_case = process_one_case(pid, CFG)
            all_case_dfs.append(df_case)
        except Exception as e:
            print(f"[ERROR] PID {pid}: {e}")

    if len(all_case_dfs) == 0:
        print("No cases were processed successfully.")
        return

    df_all = pd.concat(all_case_dfs, ignore_index=True)

    # Excel（ラベルごとタブ）
    write_label_summary_excel(
        df_all=df_all,
        out_excel=CFG["out_excel"],
        label_ids=CFG["excel_target_labels"],
        regions=("all", "core1", "core2")
    )

    print("\nDone.")
    print(df_all.head(15))


if __name__ == "__main__":
    main()