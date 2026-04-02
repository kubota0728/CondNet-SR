# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:51:06 2026

@author: kubota
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_raw_data(file_path, shape, dtype=np.float32):
    raw_data = np.fromfile(file_path, dtype=dtype)

    total_elements = np.prod(shape)
    if raw_data.size != total_elements:
        raise ValueError(
            f"データのサイズ ({raw_data.size}) が期待されるサイズ ({total_elements}) と一致しません"
        )

    # raw は (z, y, x) 順で保存されている前提で読み込む
    volume = raw_data.reshape((shape[2], shape[1], shape[0]))

    # 返り値は (x, y, z) にそろえる
    volume = np.transpose(volume, (2, 1, 0))

    return volume


def get_left_hemisphere_mask(shape, axis=0, use_first_half=True):
    """
    左脳マスクを作成
    axis=0,1,2 のどの軸で左右を分けるか
    use_first_half=True なら前半を左脳とみなす
    """
    mask = np.zeros(shape, dtype=bool)
    split = shape[axis] // 2

    slicer = [slice(None)] * 3
    if use_first_half:
        slicer[axis] = slice(0, split)
    else:
        slicer[axis] = slice(split, shape[axis])

    mask[tuple(slicer)] = True
    return mask


def compute_gradient_magnitude(volume):
    """
    勾配ノルム |∇E| を計算
    spacing は使わず、1ボクセルあたりの差分で計算
    """
    gx, gy, gz = np.gradient(volume)
    return np.sqrt(gx**2 + gy**2 + gz**2)


def compute_region_metrics(e_field, grad_mag, gx, gy, gz, region_mask):
    values = e_field[region_mask]
    grad_values = grad_mag[region_mask]
    gx_values = np.abs(gx[region_mask])
    gy_values = np.abs(gy[region_mask])
    gz_values = np.abs(gz[region_mask])

    if values.size == 0:
        return {
            "n_voxels": 0,
            "mean": np.nan,
            "std": np.nan,
            "cv_percent": np.nan,
            "p99_9": np.nan,
            "mean_grad": np.nan,
            "mean_grad_x": np.nan,
            "mean_grad_y": np.nan,
            "mean_grad_z": np.nan,
        }

    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=0))
    cv_percent = float(std_val / mean_val * 100.0) if mean_val != 0 else np.nan
    p99_9 = float(np.percentile(values, 99.9))

    return {
        "n_voxels": int(values.size),
        "mean": mean_val,
        "std": std_val,
        "cv_percent": cv_percent,
        "p99_9": p99_9,
        "mean_grad": float(np.mean(grad_values)),
        "mean_grad_x": float(np.mean(gx_values)),
        "mean_grad_y": float(np.mean(gy_values)),
        "mean_grad_z": float(np.mean(gz_values)),
    }

def compute_gradients(volume):
    """
    x, y, z 方向の勾配を返す
    """
    gx, gy, gz = np.gradient(volume)
    return gx, gy, gz

def extract_case_key_from_label(label_file):
    """
    例:
    IXI046_308_419_431.raw -> 308_419_431
    """
    stem = Path(label_file).stem
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected label filename: {label_file}")
    return "_".join(parts[1:4])


def extract_case_key_from_ef(e_file):
    """
    例:
    E_308_419_431_M1.raw -> 308_419_431
    E_308_419_431_uniform.raw -> 308_419_431
    """
    stem = Path(e_file).stem
    parts = stem.split("_")
    if len(parts) < 5:
        raise ValueError(f"Unexpected EF filename: {e_file}")
    return "_".join(parts[1:4])


def extract_condition_from_ef(e_file):
    """
    例:
    E_308_419_431_M1.raw -> M1
    E_308_419_431_uniform.raw -> uniform
    """
    stem = Path(e_file).stem
    parts = stem.split("_")
    if len(parts) < 5:
        raise ValueError(f"Unexpected EF filename: {e_file}")
    return parts[-1]


def build_label_map(label_dir):
    """
    label_dir 内のラベルファイルから
    {case_key: label_path} を作成
    """
    label_dir = Path(label_dir)
    label_map = {}

    for label_file in label_dir.glob("*.raw"):
        key = extract_case_key_from_label(label_file.name)
        if key in label_map:
            raise ValueError(f"Duplicated label key found: {key}")
        label_map[key] = label_file

    return label_map


def match_label_file(e_file, label_map):
    """
    EFファイルに対応するラベルファイルを返す
    """
    key = extract_case_key_from_ef(Path(e_file).name)
    if key not in label_map:
        raise FileNotFoundError(f"No label file found for EF case key: {key}")
    return label_map[key]


def make_display_volume(labels, left_mask, csf_labels, gm_labels, wm_labels):
    disp = np.zeros(labels.shape, dtype=np.uint8)

    csf_mask = left_mask & np.isin(labels, csf_labels)
    gm_mask = left_mask & np.isin(labels, gm_labels)
    wm_mask = left_mask & np.isin(labels, wm_labels)

    disp[csf_mask] = 1
    disp[gm_mask] = 2
    disp[wm_mask] = 3

    return disp, csf_mask, gm_mask, wm_mask


def get_nonzero_center_slices(mask):
    """
    対象領域がある範囲の中心スライスを返す
    """
    coords = np.argwhere(mask)
    if coords.size == 0:
        shape = mask.shape
        return shape[0] // 2, shape[1] // 2, shape[2] // 2

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = ((mins + maxs) // 2).astype(int)
    return int(center[0]), int(center[1]), int(center[2])


def show_left_label_volume(display_vol, case_name=""):
    """
    左脳の対象ラベルのみを3断面表示
    """
    x, y, z = get_nonzero_center_slices(display_vol > 0)

    axial = display_vol[:, :, z]
    coronal = display_vol[:, y, :]
    sagittal = display_vol[x, :, :]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    images = [axial, coronal, sagittal]
    titles = [f"Axial z={z}", f"Coronal y={y}", f"Sagittal x={x}"]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img.T, origin="lower", interpolation="nearest", vmin=0, vmax=3)
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle(f"{case_name}: left hemisphere target labels only")
    plt.tight_layout()
    plt.show()


def analyze_all_cases(
    e_dir,
    label_dir,
    shape,
    e_dtype=np.float32,
    label_dtype=np.uint8,
    csf_labels=(1,),
    gm_labels=(2,),
    wm_labels=(3,),
    hemisphere_axis=0,
    left_is_first_half=True,
    output_csv="efield_left_metrics.csv",
    output_pivot_csv="efield_left_metrics_pivot.csv",
    show_label_check=True,
):
    e_dir = Path(e_dir)
    label_dir = Path(label_dir)

    e_files = sorted(e_dir.glob("*.raw"))
    if not e_files:
        raise FileNotFoundError(f"No raw files found in {e_dir}")

    label_map = build_label_map(label_dir)
    left_mask = get_left_hemisphere_mask(
        shape=shape,
        axis=hemisphere_axis,
        use_first_half=left_is_first_half
    )

    results = []

    for e_file in e_files:
        print(f"Processing: {e_file.name}")

        label_file = match_label_file(e_file, label_map)

        e_field = load_raw_data(e_file, shape=shape, dtype=e_dtype)
        labels = load_raw_data(label_file, shape=shape, dtype=label_dtype)

        grad_mag = compute_gradient_magnitude(e_field)
        gx, gy, gz = compute_gradients(e_field)

        disp, csf_mask, gm_mask, wm_mask = make_display_volume(
            labels=labels,
            left_mask=left_mask,
            csf_labels=csf_labels,
            gm_labels=gm_labels,
            wm_labels=wm_labels,
        )

        if show_label_check:
            show_left_label_volume(disp, case_name=e_file.stem)

        total_mask = disp > 0

        csf_metrics = compute_region_metrics(e_field, grad_mag, gx, gy, gz, csf_mask)
        gm_metrics = compute_region_metrics(e_field, grad_mag, gx, gy, gz, gm_mask)
        wm_metrics = compute_region_metrics(e_field, grad_mag, gx, gy, gz, wm_mask)
        total_metrics = compute_region_metrics(e_field, grad_mag, gx, gy, gz, total_mask)

        row = {
            "ef_file": e_file.name,
            "label_file": label_file.name,
            "case_key": extract_case_key_from_ef(e_file.name),
            "condition": extract_condition_from_ef(e_file.name),

            "CSF_n": csf_metrics["n_voxels"],
            "CSF_mean": csf_metrics["mean"],
            "CSF_std": csf_metrics["std"],
            "CSF_CV_percent": csf_metrics["cv_percent"],
            "CSF_p99_9": csf_metrics["p99_9"],
            "CSF_mean_grad": csf_metrics["mean_grad"],
            "CSF_mean_grad_x": csf_metrics["mean_grad_x"],
            "CSF_mean_grad_y": csf_metrics["mean_grad_y"],
            "CSF_mean_grad_z": csf_metrics["mean_grad_z"],

            "GM_n": gm_metrics["n_voxels"],
            "GM_mean": gm_metrics["mean"],
            "GM_std": gm_metrics["std"],
            "GM_CV_percent": gm_metrics["cv_percent"],
            "GM_p99_9": gm_metrics["p99_9"],
            "GM_mean_grad": gm_metrics["mean_grad"],
            "GM_mean_grad_x": gm_metrics["mean_grad_x"],
            "GM_mean_grad_y": gm_metrics["mean_grad_y"],
            "GM_mean_grad_z": gm_metrics["mean_grad_z"],

            "WM_n": wm_metrics["n_voxels"],
            "WM_mean": wm_metrics["mean"],
            "WM_std": wm_metrics["std"],
            "WM_CV_percent": wm_metrics["cv_percent"],
            "WM_p99_9": wm_metrics["p99_9"],
            "WM_mean_grad": wm_metrics["mean_grad"],
            "WM_mean_grad_x": wm_metrics["mean_grad_x"],
            "WM_mean_grad_y": wm_metrics["mean_grad_y"],
            "WM_mean_grad_z": wm_metrics["mean_grad_z"],

            "TOTAL_n": total_metrics["n_voxels"],
            "TOTAL_mean": total_metrics["mean"],
            "TOTAL_std": total_metrics["std"],
            "TOTAL_CV_percent": total_metrics["cv_percent"],
            "TOTAL_p99_9": total_metrics["p99_9"],
            "TOTAL_mean_grad": total_metrics["mean_grad"],
            "TOTAL_mean_grad_x": total_metrics["mean_grad_x"],
            "TOTAL_mean_grad_y": total_metrics["mean_grad_y"],
            "TOTAL_mean_grad_z": total_metrics["mean_grad_z"],
        }

        results.append(row)

    df = pd.DataFrame(results)

    # 行=指標、列=モデル の形式だけ保存
    pivot_df = df.set_index("condition").T
    pivot_df.to_csv(output_csv, encoding="utf-8-sig")
    
    print(f"Saved: {output_csv}")
    return df


if __name__ == "__main__":
    # ==============================
    # ここを設定
    # ==============================

    E_DIR = r"D:\kubota\Data\Model10test\EF\IXI401"
    LABEL_DIR = r"D:\kubota\Data\Model10test\conductivity\freesurfer"


    # rawのshape
    SHAPE = (313, 423, 444)

    # dtype
    E_DTYPE = np.float32
    LABEL_DTYPE = np.float32

    # ラベル番号
    CSF_LABELS = (13,)      # 例
    GM_LABELS = (10,91) 
    WM_LABELS = (11,92)       # 例

    # 左右を分ける軸
    HEMISPHERE_AXIS = 0

    # その軸の前半が左脳ならTrue、後半が左脳ならFalse
    LEFT_IS_FIRST_HALF = True

    OUTPUT_CSV = r"D:\kubota\Data\Model10test\EF\efield_left_metrics.csv"

    # 位置確認表示をするか
    SHOW_LABEL_CHECK = True

    # ==============================
    df = analyze_all_cases(
        e_dir=E_DIR,
        label_dir=LABEL_DIR,
        shape=SHAPE,
        e_dtype=E_DTYPE,
        label_dtype=LABEL_DTYPE,
        csf_labels=CSF_LABELS,
        gm_labels=GM_LABELS,
        wm_labels=WM_LABELS,
        hemisphere_axis=HEMISPHERE_AXIS,
        left_is_first_half=LEFT_IS_FIRST_HALF,
        output_csv=OUTPUT_CSV,
        show_label_check=SHOW_LABEL_CHECK,
    )

    print(df)
    


