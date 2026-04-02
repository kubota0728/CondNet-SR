# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 13:44:52 2026

@author: kubota
"""

import os
import re
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, binary_dilation


# =========================================================
# 設定
# =========================================================
CFG = {
    # -----------------------------------------------------
    # 入力フォルダ
    # -----------------------------------------------------
    "cond_dir":  r"D:\kubota\Data\Model10test\conductivity\20260319",
    "label_dir": r"D:\kubota\Data\Model10test\conductivity\freesurfer",
    "ef_root_dir": r"D:\kubota\Data\Model10test\EF",

    # -----------------------------------------------------
    # 対象症例
    # -----------------------------------------------------
    "case_ids": [
        "IXI046", "IXI122", "IXI254", "IXI304", "IXI362",
        "IXI369", "IXI401", "IXI437", "IXI441", "IXI575"
    ],
    "model_name": "M2",   # "uniform", "M1", "M3" などに変更して使用

    # -----------------------------------------------------
    # dtype
    # -----------------------------------------------------
    "cond_dtype": np.float32,
    "label_dtype": np.float32,
    "ef_dtype": np.float32,

    # -----------------------------------------------------
    # ラベル定義
    # -----------------------------------------------------
    "csf_labels": (13,),
    "gm_labels":  (10, 91),
    "wm_labels":  (11, 92),

    # -----------------------------------------------------
    # 左右分割
    # -----------------------------------------------------
    "hemisphere_axis": 0,
    "left_is_first_half": True,

    # -----------------------------------------------------
    # shell の並び
    # -----------------------------------------------------
    "distance_mm": [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],

    # -----------------------------------------------------
    # 球 ROI
    # -----------------------------------------------------
    "surface_points_xyz": {
        "IXI046": (100, 192, 386),
        "IXI122": (96, 164, 378),
        "IXI254": (79, 189, 383),
        "IXI304": (79, 162, 364),
        "IXI362": (79, 170, 388),
        "IXI369": (101, 163, 385),
        "IXI401": (92, 153, 389),
        "IXI437": (100, 195, 372),
        "IXI441": (94, 180, 385),
        "IXI575": (99, 213, 374),
    },
    "sphere_radius_vox": 50.0,

    # -----------------------------------------------------
    # 電界指標
    # "p99"        : 各 shell 内の99 percentile
    # "top10_mean" : 各 shell 内の上位10%平均
    # -----------------------------------------------------
    "ef_metric_mode": "top10_mean",

    # -----------------------------------------------------
    # 勾配指標
    # grad_top_percent = 10.0 -> 上位10%平均
    # -----------------------------------------------------
    "grad_top_percent": 10.0,

    # -----------------------------------------------------
    # 出力
    # -----------------------------------------------------
    "output_dir": r"D:\kubota\Data\Model10test\profiles_local_sphere_grad",
    "show_plot_each_case": True,
    "save_plot_each_case": True,
    "show_plot_mean": True,
    "save_plot_mean": True,

    # -----------------------------------------------------
    # デバッグ表示
    # -----------------------------------------------------
    "show_shell_check": False,
    "show_sphere_check": False,
}


# =========================================================
# 基本関数
# =========================================================
def ensure_parent_dir(path):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    return float(np.mean(values)), float(np.std(values, ddof=0))


def extract_case_and_shape_from_cond_filename(filepath, model_name):
    base = os.path.basename(filepath)
    pattern = rf"^(IXI\d{{3}})_(\d+)_(\d+)_(\d+)_{re.escape(model_name)}\.raw$"
    m = re.match(pattern, base)
    if m is None:
        raise ValueError(f"cond filename の形式が想定と異なります: {base}")

    case_id = m.group(1)
    shape = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
    return case_id, shape


def find_case_files(case_id, model_name, cond_dir, label_dir, ef_root_dir):
    cond_pattern = os.path.join(cond_dir, f"{case_id}_*_*_*_{model_name}.raw")
    cond_candidates = glob.glob(cond_pattern)

    if len(cond_candidates) == 0:
        raise FileNotFoundError(f"cond ファイルが見つかりません: {cond_pattern}")
    if len(cond_candidates) > 1:
        raise RuntimeError(f"cond ファイルが複数見つかりました: {cond_candidates}")

    cond_path = cond_candidates[0]
    _, shape = extract_case_and_shape_from_cond_filename(cond_path, model_name)
    sx, sy, sz = shape

    label_path = os.path.join(label_dir, f"{case_id}_{sx}_{sy}_{sz}.raw")
    ef_path = os.path.join(ef_root_dir, case_id, f"E_{sx}_{sy}_{sz}_{model_name}.raw")

    if not os.path.exists(label_path):
        raise FileNotFoundError(f"label ファイルが見つかりません: {label_path}")
    if not os.path.exists(ef_path):
        raise FileNotFoundError(f"ef ファイルが見つかりません: {ef_path}")

    return {
        "case_id": case_id,
        "shape": shape,
        "cond_raw_path": cond_path,
        "label_raw_path": label_path,
        "ef_raw_path": ef_path,
    }


def load_raw_data(file_path, shape, dtype=np.float32):
    raw_data = np.fromfile(file_path, dtype=dtype)
    total_elements = np.prod(shape)

    if raw_data.size != total_elements:
        raise ValueError(
            f"データサイズ不一致: {file_path}\n"
            f"read={raw_data.size}, expected={total_elements}"
        )

    # raw は (z, y, x) 順と仮定
    volume = raw_data.reshape((shape[2], shape[1], shape[0]))
    volume = np.transpose(volume, (2, 1, 0))  # -> (x, y, z)
    return volume


def get_left_hemisphere_mask(shape, axis=0, use_first_half=True):
    mask = np.zeros(shape, dtype=bool)
    split = shape[axis] // 2

    slicer = [slice(None)] * 3
    slicer[axis] = slice(0, split) if use_first_half else slice(split, shape[axis])
    mask[tuple(slicer)] = True
    return mask


def get_surface_point_for_case(case_id, cfg):
    surface_points = cfg.get("surface_points_xyz", {})
    if case_id not in surface_points:
        raise KeyError(f"surface_points_xyz に {case_id} の座標がありません。")
    return np.asarray(surface_points[case_id], dtype=np.float64)


# =========================================================
# shell 作成
# =========================================================
def build_brain_shells_from_csf(labels, left_mask, csf_labels, gm_labels, wm_labels):
    structure = np.ones((3, 3, 3), dtype=bool)

    csf_mask = left_mask & np.isin(labels, csf_labels)
    gm_mask  = left_mask & np.isin(labels, gm_labels)
    wm_mask  = left_mask & np.isin(labels, wm_labels)

    brain_mask = gm_mask | wm_mask

    if not np.any(brain_mask):
        raise ValueError("左脳内に GM+WM が見つかりませんでした。")
    if not np.any(csf_mask):
        raise ValueError("左脳内に CSF が見つかりませんでした。")

    shells = {}

    csf_dil = binary_dilation(csf_mask, structure=structure, iterations=1)
    brain_boundary = brain_mask & csf_dil

    if not np.any(brain_boundary):
        raise ValueError("(GM+WM)-CSF boundary が見つかりませんでした。")

    current = brain_mask.copy()
    eroded = binary_erosion(current, structure=structure, iterations=1, border_value=0)
    outer_shell = current & (~eroded)

    shells[0.0] = outer_shell & brain_boundary
    current = eroded

    for i in range(1, 7):
        d_mm = 0.5 * i
        eroded = binary_erosion(current, structure=structure, iterations=1, border_value=0)
        shells[d_mm] = current & (~eroded)
        current = eroded

    brain_dil = binary_dilation(brain_mask, structure=structure, iterations=1)
    shell_m05 = csf_mask & brain_dil
    shells[-0.5] = shell_m05

    csf_remaining = csf_mask & (~shell_m05)
    shell_m10 = csf_remaining & binary_dilation(shell_m05, structure=structure, iterations=1)
    shells[-1.0] = shell_m10

    return shells, brain_mask, brain_boundary


# =========================================================
# 球 ROI
# =========================================================
def create_sphere_mask(shape, center_xyz, radius_vox):
    center_xyz = np.asarray(center_xyz, dtype=np.float64)
    cx, cy, cz = center_xyz

    r = float(radius_vox)
    margin = int(np.ceil(r)) + 1

    x0 = max(0, int(np.floor(cx - margin)))
    x1 = min(shape[0], int(np.ceil(cx + margin + 1)))
    y0 = max(0, int(np.floor(cy - margin)))
    y1 = min(shape[1], int(np.ceil(cy + margin + 1)))
    z0 = max(0, int(np.floor(cz - margin)))
    z1 = min(shape[2], int(np.ceil(cz + margin + 1)))

    xs = np.arange(x0, x1, dtype=np.float64)
    ys = np.arange(y0, y1, dtype=np.float64)
    zs = np.arange(z0, z1, dtype=np.float64)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    d2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
    local_mask = d2 <= (r ** 2)

    mask = np.zeros(shape, dtype=bool)
    mask[x0:x1, y0:y1, z0:z1] = local_mask
    return mask


# =========================================================
# 上位平均
# =========================================================
def compute_top_fraction_mean(values, top_percent):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan

    q = 100.0 - float(top_percent)
    thr = np.percentile(values, q)
    top_vals = values[values >= thr]

    if top_vals.size == 0:
        return np.nan
    return float(np.mean(top_vals))


# =========================================================
# 集計
# =========================================================
def summarize_shells_local_sphere_with_gradient(
    cond,
    labels,
    efield,
    left_mask,
    shells,
    sphere_mask,
    csf_labels,
    gm_labels,
    wm_labels,
    ef_metric_mode="top10_mean",
    grad_top_percent=10.0,
):
    # -----------------------------
    # 左脳全体で電界を正規化
    # -----------------------------
    left_valid_mask = left_mask & (labels > 0)

    ef_base = efield[left_valid_mask]
    ef_base = ef_base[np.isfinite(ef_base)]
    if ef_base.size == 0:
        raise ValueError("左脳(label>0) 内に有効な電界値がありません。")

    ef_p99_9 = float(np.percentile(ef_base.ravel(), 99.9))
    if ef_p99_9 == 0:
        raise ValueError("左脳99.9 percentile が 0 です。")

    efield_clipped = np.minimum(efield, ef_p99_9)
    efield_norm = efield_clipped / ef_p99_9

    # -----------------------------
    # 勾配 |∇E|
    # -----------------------------
    gx, gy, gz = np.gradient(efield_norm)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)

    csf_set = np.isin(labels, csf_labels)
    gm_set = np.isin(labels, gm_labels)
    wm_set = np.isin(labels, wm_labels)

    rows = []

    for dist in sorted(shells.keys()):
        mask = shells[dist] & sphere_mask
        n = int(np.count_nonzero(mask))

        if n == 0:
            rows.append({
                "distance_mm": dist,
                "n_voxels": 0,
                "cond_mean": np.nan,
                "cond_std": np.nan,
                "csf_ratio": np.nan,
                "gm_ratio": np.nan,
                "wm_ratio": np.nan,
                "other_ratio": np.nan,
                "ef_metric_value": np.nan,
                "ef_norm_std": np.nan,
                "grad_mean": np.nan,
                "grad_std": np.nan,
                "grad_top_mean": np.nan,
                "ef_norm_p99_9_leftbrain": ef_p99_9,
            })
            continue

        cond_vals = cond[mask]
        ef_vals = efield_norm[mask]
        grad_vals = grad_mag[mask]

        cond_mean, cond_std = safe_mean_std(cond_vals)
        _, ef_std = safe_mean_std(ef_vals)
        grad_mean, grad_std = safe_mean_std(grad_vals)

        if ef_vals.size == 0:
            ef_metric_value = np.nan
        else:
            if ef_metric_mode == "p99":
                ef_metric_value = float(np.percentile(ef_vals, 99))
            elif ef_metric_mode == "top10_mean":
                ef_metric_value = compute_top_fraction_mean(ef_vals, top_percent=10.0)
            else:
                raise ValueError(f"未知の ef_metric_mode: {ef_metric_mode}")

        grad_top_mean = compute_top_fraction_mean(grad_vals, top_percent=grad_top_percent)

        csf_ratio = float(np.count_nonzero(mask & csf_set) / n)
        gm_ratio = float(np.count_nonzero(mask & gm_set) / n)
        wm_ratio = float(np.count_nonzero(mask & wm_set) / n)
        other_ratio = float(1.0 - csf_ratio - gm_ratio - wm_ratio)

        rows.append({
            "distance_mm": dist,
            "n_voxels": n,
            "cond_mean": cond_mean,
            "cond_std": cond_std,
            "csf_ratio": csf_ratio,
            "gm_ratio": gm_ratio,
            "wm_ratio": wm_ratio,
            "other_ratio": other_ratio,
            "ef_metric_value": ef_metric_value,
            "ef_norm_std": ef_std,
            "grad_mean": grad_mean,
            "grad_std": grad_std,
            "grad_top_mean": grad_top_mean,
            "ef_norm_p99_9_leftbrain": ef_p99_9,
        })

    return pd.DataFrame(rows)


def aggregate_case_dfs(case_dfs, distance_order):
    all_df = pd.concat(case_dfs, ignore_index=True)

    records = []
    for dist in distance_order:
        sub = all_df[all_df["distance_mm"] == dist]

        row = {"distance_mm": dist, "n_cases": sub["case_id"].nunique()}

        for col in [
            "n_voxels",
            "cond_mean",
            "cond_std",
            "csf_ratio",
            "gm_ratio",
            "wm_ratio",
            "other_ratio",
            "ef_metric_value",
            "ef_norm_std",
            "grad_mean",
            "grad_std",
            "grad_top_mean",
            "ef_norm_p99_9_leftbrain",
        ]:
            vals = sub[col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]

            if vals.size == 0:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_std"] = np.nan
            else:
                row[f"{col}_mean"] = float(np.mean(vals))
                row[f"{col}_std"] = float(np.std(vals, ddof=0))

        records.append(row)

    return pd.DataFrame(records), all_df


# =========================================================
# ラベル・球ROI確認表示
# =========================================================
def show_shell_check(labels, shells, brain_boundary, csf_labels, gm_labels, wm_labels):
    z = labels.shape[2] // 2

    disp = np.zeros(labels.shape, dtype=np.uint8)
    disp[np.isin(labels, csf_labels)] = 1
    disp[np.isin(labels, gm_labels)] = 2
    disp[np.isin(labels, wm_labels)] = 3
    disp[brain_boundary] = 4

    for i, dist in enumerate(sorted(shells.keys()), start=5):
        disp[shells[dist]] = i

    plt.figure(figsize=(6, 6))
    plt.imshow(disp[:, :, z].T, origin="lower", interpolation="nearest")
    plt.title(f"Shell check (axial z={z})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_sphere_check(labels, sphere_mask, surface_point_xyz, csf_labels, gm_labels, wm_labels):
    disp = np.zeros(labels.shape, dtype=np.uint8)
    disp[np.isin(labels, csf_labels)] = 1
    disp[np.isin(labels, gm_labels)] = 2
    disp[np.isin(labels, wm_labels)] = 3

    x, y, z = np.rint(surface_point_xyz).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    img = disp[x, :, :].T
    ax.imshow(img, origin="lower", interpolation="nearest")
    yy, zz = np.where(sphere_mask[x, :, :].T)
    if len(yy) > 0:
        ax.scatter(yy, zz, s=1, c="white", alpha=0.3)
    ax.scatter(y, z, s=80, facecolors="none", edgecolors="red", linewidths=2)
    ax.set_title(f"Sagittal (x={x})")
    ax.set_xlabel("y")
    ax.set_ylabel("z")

    ax = axes[1]
    img = disp[:, y, :].T
    ax.imshow(img, origin="lower", interpolation="nearest")
    xx, zz = np.where(sphere_mask[:, y, :].T)
    if len(xx) > 0:
        ax.scatter(xx, zz, s=1, c="white", alpha=0.3)
    ax.scatter(x, z, s=80, facecolors="none", edgecolors="red", linewidths=2)
    ax.set_title(f"Coronal (y={y})")
    ax.set_xlabel("x")
    ax.set_ylabel("z")

    ax = axes[2]
    img = disp[:, :, z].T
    yy, xx = np.where(sphere_mask[:, :, z].T)
    ax.imshow(img, origin="lower", interpolation="nearest")
    if len(xx) > 0:
        ax.scatter(xx, yy, s=1, c="white", alpha=0.3)
    ax.scatter(x, y, s=80, facecolors="none", edgecolors="red", linewidths=2)
    ax.set_title(f"Axial (z={z})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.tight_layout()
    plt.show()


# =========================================================
# プロット
# =========================================================
def get_ef_metric_label(cfg):
    if cfg["ef_metric_mode"] == "p99":
        return "99th percentile normalized E-field"
    elif cfg["ef_metric_mode"] == "top10_mean":
        return "Top 10% mean normalized E-field"
    return "Normalized E-field"


def plot_profiles_single_case(df, case_id, model_name, cfg, output_fig=None, show_plot=True):
    x = df["distance_mm"].to_numpy()
    ef_label = get_ef_metric_label(cfg)

    fig, axes = plt.subplots(
        5, 1,
        figsize=(8, 14),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 0.7, 1.2, 1.0, 1.0]}
    )

    # Conductivity
    ax = axes[0]
    y = df["cond_mean"].to_numpy()
    sd = df["cond_std"].to_numpy()
    ax.plot(x, y, marker="o", linewidth=2, label="Conductivity mean")
    ax.fill_between(x, y - sd, y + sd, alpha=0.2, label="±SD")
    ax.set_ylabel("Conductivity")
    ax.set_title(f"{case_id} | {model_name} | Local sphere shell profile with gradient")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Label composition
    ax = axes[1]
    csf = df["csf_ratio"].to_numpy()
    gm = df["gm_ratio"].to_numpy()
    wm = df["wm_ratio"].to_numpy()

    cum1 = csf
    cum2 = csf + gm
    cum3 = csf + gm + wm

    ax.fill_between(x, 0, cum1, step="mid", alpha=0.8, label="CSF")
    ax.fill_between(x, cum1, cum2, step="mid", alpha=0.8, label="GM")
    ax.fill_between(x, cum2, cum3, step="mid", alpha=0.8, label="WM")
    ax.set_ylabel("Ratio")
    ax.set_ylim(0, 1.0)
    ax.set_title("Label composition")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # E-field metric
    ax = axes[2]
    y = df["ef_metric_value"].to_numpy()
    sd = df["ef_norm_std"].to_numpy()
    ax.plot(x, y, marker="o", linewidth=2, label=ef_label)
    ax.fill_between(x, y - sd, y + sd, alpha=0.2, label="±SD")
    ax.set_ylabel("Normalized E-field")
    ax.set_title("E-field metric")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Gradient mean
    ax = axes[3]
    y = df["grad_mean"].to_numpy()
    sd = df["grad_std"].to_numpy()
    ax.plot(x, y, marker="o", linewidth=2, label="Mean |∇E|")
    ax.fill_between(x, y - sd, y + sd, alpha=0.2, label="±SD")
    ax.set_ylabel("|∇E|")
    ax.set_title("Mean gradient magnitude")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Gradient top mean
    ax = axes[4]
    y = df["grad_top_mean"].to_numpy()
    ax.plot(x, y, marker="o", linewidth=2, label=f"Top {cfg['grad_top_percent']}% mean |∇E|")
    ax.set_ylabel("|∇E|")
    ax.set_xlabel("Shell position (mm)")
    ax.set_title(f"Top {cfg['grad_top_percent']}% mean gradient magnitude")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax.set_xticks(x)
    plt.tight_layout()

    if output_fig is not None:
        ensure_parent_dir(output_fig)
        plt.savefig(output_fig, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_profiles_mean(df_mean, model_name, cfg, output_fig=None, show_plot=True):
    x = df_mean["distance_mm"].to_numpy()
    ef_label = get_ef_metric_label(cfg)

    fig, axes = plt.subplots(
        5, 1,
        figsize=(8, 14),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 0.7, 1.2, 1.0, 1.0]}
    )

    # Conductivity
    ax = axes[0]
    y = df_mean["cond_mean_mean"].to_numpy()
    sd = df_mean["cond_mean_std"].to_numpy()
    ax.plot(x, y, marker="o", linewidth=2, label="Mean across cases")
    ax.fill_between(x, y - sd, y + sd, alpha=0.2, label="±SD across cases")
    ax.set_ylabel("Conductivity")
    ax.set_title(f"All cases mean | {model_name} | Local sphere shell profile with gradient")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Label composition
    ax = axes[1]
    csf = df_mean["csf_ratio_mean"].to_numpy()
    gm = df_mean["gm_ratio_mean"].to_numpy()
    wm = df_mean["wm_ratio_mean"].to_numpy()

    cum1 = csf
    cum2 = csf + gm
    cum3 = csf + gm + wm

    ax.fill_between(x, 0, cum1, step="mid", alpha=0.8, label="CSF")
    ax.fill_between(x, cum1, cum2, step="mid", alpha=0.8, label="GM")
    ax.fill_between(x, cum2, cum3, step="mid", alpha=0.8, label="WM")
    ax.set_ylabel("Ratio")
    ax.set_ylim(0, 1.0)
    ax.set_title("Mean label composition across cases")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # E-field metric
    ax = axes[2]
    y = df_mean["ef_metric_value_mean"].to_numpy()
    sd = df_mean["ef_metric_value_std"].to_numpy()
    ax.plot(x, y, marker="o", linewidth=2, label=ef_label)
    ax.fill_between(x, y - sd, y + sd, alpha=0.2, label="±SD across cases")
    ax.set_ylabel("Normalized E-field")
    ax.set_title("Mean E-field metric across cases")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Gradient mean
    ax = axes[3]
    y = df_mean["grad_mean_mean"].to_numpy()
    sd = df_mean["grad_mean_std"].to_numpy()
    ax.plot(x, y, marker="o", linewidth=2, label="Mean |∇E|")
    ax.fill_between(x, y - sd, y + sd, alpha=0.2, label="±SD across cases")
    ax.set_ylabel("|∇E|")
    ax.set_title("Mean gradient magnitude across cases")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Gradient top mean
    ax = axes[4]
    y = df_mean["grad_top_mean_mean"].to_numpy()
    sd = df_mean["grad_top_mean_std"].to_numpy()
    ax.plot(x, y, marker="o", linewidth=2, label=f"Top {cfg['grad_top_percent']}% mean |∇E|")
    ax.fill_between(x, y - sd, y + sd, alpha=0.2, label="±SD across cases")
    ax.set_ylabel("|∇E|")
    ax.set_xlabel("Shell position (mm)")
    ax.set_title(f"Top {cfg['grad_top_percent']}% mean gradient magnitude across cases")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax.set_xticks(x)
    plt.tight_layout()

    if output_fig is not None:
        ensure_parent_dir(output_fig)
        plt.savefig(output_fig, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# =========================================================
# 症例ごとの処理
# =========================================================
def process_one_case(case_info, cfg):
    case_id = case_info["case_id"]
    shape = case_info["shape"]

    cond = load_raw_data(case_info["cond_raw_path"], shape, dtype=cfg["cond_dtype"]).astype(np.float32)
    labels = load_raw_data(case_info["label_raw_path"], shape, dtype=cfg["label_dtype"])
    efield = load_raw_data(case_info["ef_raw_path"], shape, dtype=cfg["ef_dtype"]).astype(np.float32)

    labels = np.rint(labels).astype(np.int32)

    left_mask = get_left_hemisphere_mask(
        shape=shape,
        axis=cfg["hemisphere_axis"],
        use_first_half=cfg["left_is_first_half"]
    )

    shells, brain_mask, brain_boundary = build_brain_shells_from_csf(
        labels=labels,
        left_mask=left_mask,
        csf_labels=cfg["csf_labels"],
        gm_labels=cfg["gm_labels"],
        wm_labels=cfg["wm_labels"],
    )

    surface_point_xyz = get_surface_point_for_case(case_id, cfg)

    sphere_mask = create_sphere_mask(
        shape=shape,
        center_xyz=surface_point_xyz,
        radius_vox=cfg["sphere_radius_vox"],
    )
    sphere_mask &= left_mask

    if cfg.get("show_shell_check", False):
        show_shell_check(
            labels=labels,
            shells=shells,
            brain_boundary=brain_boundary,
            csf_labels=cfg["csf_labels"],
            gm_labels=cfg["gm_labels"],
            wm_labels=cfg["wm_labels"],
        )

    if cfg.get("show_sphere_check", False):
        show_sphere_check(
            labels=labels,
            sphere_mask=sphere_mask,
            surface_point_xyz=surface_point_xyz,
            csf_labels=cfg["csf_labels"],
            gm_labels=cfg["gm_labels"],
            wm_labels=cfg["wm_labels"],
        )

    df = summarize_shells_local_sphere_with_gradient(
        cond=cond,
        labels=labels,
        efield=efield,
        left_mask=left_mask,
        shells=shells,
        sphere_mask=sphere_mask,
        csf_labels=cfg["csf_labels"],
        gm_labels=cfg["gm_labels"],
        wm_labels=cfg["wm_labels"],
        ef_metric_mode=cfg["ef_metric_mode"],
        grad_top_percent=cfg["grad_top_percent"],
    )

    df = df.set_index("distance_mm").reindex(cfg["distance_mm"]).reset_index()

    df.insert(0, "case_id", case_id)
    df.insert(1, "model_name", cfg["model_name"])
    df.insert(2, "shape_x", shape[0])
    df.insert(3, "shape_y", shape[1])
    df.insert(4, "shape_z", shape[2])

    df["surface_x"] = surface_point_xyz[0]
    df["surface_y"] = surface_point_xyz[1]
    df["surface_z"] = surface_point_xyz[2]
    df["sphere_radius_vox"] = cfg["sphere_radius_vox"]
    df["ef_metric_mode"] = cfg["ef_metric_mode"]
    df["grad_top_percent"] = cfg["grad_top_percent"]

    return df


# =========================================================
# メイン
# =========================================================
def main(cfg):
    model_name = cfg["model_name"]

    mode_suffix = (
        f"sphereR{cfg['sphere_radius_vox']}_"
        f"{cfg['ef_metric_mode']}_"
        f"gradTop{cfg['grad_top_percent']}"
    )
    output_dir = os.path.join(cfg["output_dir"], f"{model_name}_{mode_suffix}")
    ensure_dir(output_dir)

    case_ids = cfg["case_ids"]
    if len(case_ids) == 0:
        raise ValueError("対象症例がありません。")

    print("対象症例:", case_ids)
    print("model_name:", model_name)
    print("sphere_radius_vox:", cfg["sphere_radius_vox"])
    print("ef_metric_mode:", cfg["ef_metric_mode"])
    print("grad_top_percent:", cfg["grad_top_percent"])

    case_dfs = []

    for case_id in case_ids:
        print(f"\n===== Processing {case_id} | {model_name} =====")

        case_info = find_case_files(
            case_id=case_id,
            model_name=model_name,
            cond_dir=cfg["cond_dir"],
            label_dir=cfg["label_dir"],
            ef_root_dir=cfg["ef_root_dir"],
        )

        print("cond :", case_info["cond_raw_path"])
        print("label:", case_info["label_raw_path"])
        print("ef   :", case_info["ef_raw_path"])
        print("shape:", case_info["shape"])

        df_case = process_one_case(case_info, cfg)
        case_dfs.append(df_case)

        case_csv = os.path.join(output_dir, f"{case_id}_{model_name}_local_sphere_shell_profile_grad.csv")
        df_case.to_csv(case_csv, index=False, encoding="utf-8-sig", float_format="%.8f")
        print(f"Saved CSV: {case_csv}")
        print(df_case)

        case_fig = os.path.join(output_dir, f"{case_id}_{model_name}_local_sphere_shell_profile_grad.png")
        plot_profiles_single_case(
            df_case,
            case_id=case_id,
            model_name=model_name,
            cfg=cfg,
            output_fig=case_fig if cfg["save_plot_each_case"] else None,
            show_plot=cfg["show_plot_each_case"],
        )

    df_mean, df_all = aggregate_case_dfs(case_dfs, cfg["distance_mm"])

    all_csv = os.path.join(output_dir, f"all_cases_{model_name}_local_sphere_shell_profiles_grad_long.csv")
    mean_csv = os.path.join(output_dir, f"all_cases_{model_name}_local_sphere_shell_profile_grad_mean_sd.csv")

    df_all.to_csv(all_csv, index=False, encoding="utf-8-sig", float_format="%.8f")
    df_mean.to_csv(mean_csv, index=False, encoding="utf-8-sig", float_format="%.8f")

    print(f"\nSaved all-case long CSV : {all_csv}")
    print(f"Saved mean/std CSV      : {mean_csv}")

    print("\n===== All cases long table =====")
    print(df_all)

    print("\n===== All cases mean ± SD table =====")
    print(df_mean)

    mean_fig = os.path.join(output_dir, f"all_cases_{model_name}_local_sphere_shell_profile_grad_mean_sd.png")
    plot_profiles_mean(
        df_mean,
        model_name=model_name,
        cfg=cfg,
        output_fig=mean_fig if cfg["save_plot_mean"] else None,
        show_plot=cfg["show_plot_mean"],
    )

    return df_all, df_mean


if __name__ == "__main__":
    df_all_result, df_mean_result = main(CFG)