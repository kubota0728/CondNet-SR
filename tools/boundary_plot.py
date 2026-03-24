# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 11:14:42 2026

@author: kubota
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import binary_dilation


# ============================================================
# ✅ ここだけ編集すればOK
# ============================================================
CFG = {
    "pids": [46, 122, 254, 304, 362, 369, 401, 437, 441, 575],

    "cond_nii_tmpl":  "D:/kubota/Data/Model10test/M3_logmae099_099_04_nii/{pid}_pred.nii.gz",
    "label_nii_tmpl": "D:/kubota/Data/Model9/label14/IXI{pid:03d}_label_after.nii.gz",

    "out_dir": "D:/Kubota/data/Model10test/gm_wm_boundary_analysis/M3_logmae099_099_04",

    # ラベル定義
    "GM_LABELS": [8],
    "WM_LABELS": [13],

    # 図
    "show_figures": False,
    "save_figures": True,
    "figure_name": "gm_wm_boundary_violin.png",
    "violin_max_points_per_region": 3000,

    # 近傍定義
    "connectivity_structure": np.ones((3, 3, 3), dtype=bool),
}
# ============================================================


REGION_ORDER = ["gm_m2", "gm_m1", "boundary", "wm_p1", "wm_p2"]
REGION_LABEL_MAP = {
    "gm_m2": "GM-2",
    "gm_m1": "GM-1",
    "boundary": "Boundary",
    "wm_p1": "WM+1",
    "wm_p2": "WM+2",
}


def load_nii_img(path: str) -> nib.Nifti1Image:
    return nib.load(path)


def format_path(template: str, pid: int) -> str:
    return template.format(pid=pid)


def stats_from_values(vals: np.ndarray):
    if vals.size == 0:
        return np.nan, np.nan, np.nan, 0
    mean_v = float(np.mean(vals))
    std_v = float(np.std(vals, ddof=0))
    cv_v = float(std_v / mean_v) if mean_v != 0 else np.nan
    return mean_v, std_v, cv_v, int(vals.size)


def get_gm_wm_masks(label_zyx: np.ndarray, gm_labels, wm_labels):
    gm_mask = np.isin(label_zyx, gm_labels)
    wm_mask = np.isin(label_zyx, wm_labels)
    return gm_mask, wm_mask


def make_boundary_shell_masks(label_zyx: np.ndarray, gm_labels, wm_labels, structure=None):
    """
    boundary の定義:
        GM と接している WM ボクセル
    出力:
        gm_m2, gm_m1, boundary, wm_p1, wm_p2
    """
    if structure is None:
        structure = np.ones((3, 3, 3), dtype=bool)

    gm_mask, wm_mask = get_gm_wm_masks(label_zyx, gm_labels, wm_labels)

    if not np.any(gm_mask):
        raise ValueError("GM_LABELS に対応するボクセルが存在しません。")
    if not np.any(wm_mask):
        raise ValueError("WM_LABELS に対応するボクセルが存在しません。")

    # boundary = GM と接する WM
    gm_dil = binary_dilation(gm_mask, structure=structure)
    boundary = wm_mask & gm_dil

    # GM側 1層
    wm_dil = binary_dilation(wm_mask, structure=structure)
    gm_m1 = gm_mask & wm_dil

    # GM側 2層
    gm_m1_dil = binary_dilation(gm_m1, structure=structure)
    gm_m2 = gm_mask & gm_m1_dil & (~gm_m1)

    # WM側 +1
    boundary_dil = binary_dilation(boundary, structure=structure)
    wm_p1 = wm_mask & boundary_dil & (~boundary)

    # WM側 +2
    wm_p1_dil = binary_dilation(wm_p1, structure=structure)
    wm_p2 = wm_mask & wm_p1_dil & (~boundary) & (~wm_p1)

    masks = {
        "gm_m2": gm_m2,
        "gm_m1": gm_m1,
        "boundary": boundary,
        "wm_p1": wm_p1,
        "wm_p2": wm_p2,
    }

    return masks


def make_wide_voxel_table(cond_zyx: np.ndarray, masks: dict) -> pd.DataFrame:
    """
    5列だけの全ボクセル表を作る
    列: gm_m2, gm_m1, boundary, wm_p1, wm_p2
    """
    cols = {}
    max_len = 0

    for region in REGION_ORDER:
        vals = cond_zyx[masks[region]].astype(np.float32)
        cols[region] = vals
        max_len = max(max_len, len(vals))

    out = {}
    for region in REGION_ORDER:
        vals = cols[region]
        arr = np.full(max_len, np.nan, dtype=np.float32)
        arr[:len(vals)] = vals
        out[region] = arr

    return pd.DataFrame(out)


def make_case_mean_row(pid: int, cond_zyx: np.ndarray, masks: dict) -> pd.DataFrame:
    """
    1症例ぶんの平均・SD・CV・n_vox
    横持ちではなく、あとで集計しやすい long 形式
    """
    rows = []
    for region in REGION_ORDER:
        vals = cond_zyx[masks[region]]
        mean_v, std_v, cv_v, n_v = stats_from_values(vals)
        rows.append({
            "pid": pid,
            "region": region,
            "cond_mean": mean_v,
            "cond_std": std_v,
            "cond_cv": cv_v,
            "n_vox": n_v,
        })
    return pd.DataFrame(rows)


def make_case_mean_row_wide(pid: int, cond_zyx: np.ndarray, masks: dict) -> pd.DataFrame:
    """
    1症例ぶんの平均を5列で保存するための表
    """
    row = {"pid": pid}
    for region in REGION_ORDER:
        vals = cond_zyx[masks[region]]
        mean_v, _, _, _ = stats_from_values(vals)
        row[region] = mean_v
    return pd.DataFrame([row])


def process_one_case(pid: int, cfg: dict):
    print(f"\n===== Processing PID {pid} =====")

    cond_path  = format_path(cfg["cond_nii_tmpl"], pid)
    label_path = format_path(cfg["label_nii_tmpl"], pid)

    for p in [cond_path, label_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")

    cond_img = load_nii_img(cond_path)
    label_img = load_nii_img(label_path)

    cond = cond_img.get_fdata().astype(np.float32)
    lab = label_img.get_fdata()
    lab_int = np.rint(lab).astype(np.int32)

    # cond を label に合わせる
    cond_aligned = np.transpose(cond, (1, 0, 2))

    if cond_aligned.shape != lab_int.shape:
        raise ValueError(f"Shape mismatch: cond{cond_aligned.shape} vs label{lab_int.shape}")

    masks = make_boundary_shell_masks(
        label_zyx=lab_int,
        gm_labels=cfg["GM_LABELS"],
        wm_labels=cfg["WM_LABELS"],
        structure=cfg["connectivity_structure"],
    )

    # 件数確認
    print(f"PID {pid} voxel counts:")
    for name in REGION_ORDER:
        print(f"  {name}: {int(np.count_nonzero(masks[name]))}")

    df_voxel_wide = make_wide_voxel_table(cond_aligned, masks)
    df_case_mean_long = make_case_mean_row(pid, cond_aligned, masks)
    df_case_mean_wide = make_case_mean_row_wide(pid, cond_aligned, masks)

    return df_voxel_wide, df_case_mean_long, df_case_mean_wide


def plot_violin_from_wide_tables(case_voxel_tables: list, cfg: dict):
    """
    症例ごとの5列表を結合して描画
    """
    if len(case_voxel_tables) == 0:
        print("No data for violin plot.")
        return

    sampled_frames = []
    n_plot = int(cfg.get("violin_max_points_per_region", 3000))

    for df_case in case_voxel_tables:
        for region in REGION_ORDER:
            vals = df_case[region].dropna()
            if len(vals) == 0:
                continue

            if len(vals) > n_plot:
                vals = vals.sample(n=n_plot, random_state=0)

            sampled_frames.append(pd.DataFrame({
                "region": region,
                "conductivity": vals.to_numpy(dtype=np.float32)
            }))

    if len(sampled_frames) == 0:
        print("No sampled data for plot.")
        return

    df_plot = pd.concat(sampled_frames, ignore_index=True)

    print("Violin plot sample counts:")
    print(df_plot["region"].value_counts())

    data_list = [
        df_plot.loc[df_plot["region"] == r, "conductivity"].to_numpy()
        for r in REGION_ORDER
        if np.any(df_plot["region"] == r)
    ]
    labels_present = [r for r in REGION_ORDER if np.any(df_plot["region"] == r)]
    positions = np.arange(1, len(data_list) + 1)

    fig, ax = plt.subplots(figsize=(9, 6))

    vp = ax.violinplot(
        data_list,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    for body in vp["bodies"]:
        body.set_alpha(0.6)

    ax.boxplot(
        data_list,
        positions=positions,
        widths=0.15,
        patch_artist=True,
        showfliers=False
    )

    ax.set_xticks(positions)
    ax.set_xticklabels([REGION_LABEL_MAP[r] for r in labels_present])
    ax.set_xlabel("Region")
    ax.set_ylabel("Conductivity")
    ax.set_title("Conductivity around GM-WM boundary")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 0.25)

    plt.tight_layout()

    if cfg.get("save_figures", False):
        out_path = os.path.join(cfg["out_dir"], cfg["figure_name"])
        os.makedirs(cfg["out_dir"], exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure: {out_path}")

    if cfg.get("show_figures", False):
        plt.show()
    else:
        plt.close(fig)


def save_outputs(case_voxel_tables: list, df_case_means_long: pd.DataFrame, df_case_means_wide: pd.DataFrame, cfg: dict):
    os.makedirs(cfg["out_dir"], exist_ok=True)

    # 症例ごとの全ボクセルCSV
    for pid, df_vox in zip(cfg["pids"], case_voxel_tables):
        out_case_csv = os.path.join(cfg["out_dir"], f"boundary_voxels_pid_{pid}.csv")
        df_vox.to_csv(out_case_csv, index=False, float_format="%.8f")
        print(f"Saved voxel CSV : {out_case_csv}")

    # 症例平均 5列
    out_mean_wide_csv = os.path.join(cfg["out_dir"], "boundary_case_means_wide.csv")
    df_case_means_wide.to_csv(out_mean_wide_csv, index=False, float_format="%.8f")
    print(f"Saved mean CSV  : {out_mean_wide_csv}")

    # long 形式も保存
    out_mean_long_csv = os.path.join(cfg["out_dir"], "boundary_case_means_long.csv")
    df_case_means_long.to_csv(out_mean_long_csv, index=False, float_format="%.8f")
    print(f"Saved mean CSV  : {out_mean_long_csv}")

    # summary
    df_summary = (
        df_case_means_long
        .groupby("region", dropna=False)
        .agg(
            cond_mean_mean=("cond_mean", "mean"),
            cond_std_mean=("cond_std", "mean"),
            cond_cv_mean=("cond_cv", "mean"),
            n_vox_mean=("n_vox", "mean"),
            n_cases=("pid", "nunique"),
        )
        .reset_index()
    )

    out_excel = os.path.join(cfg["out_dir"], "boundary_summary.xlsx")
    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        df_case_means_wide.to_excel(writer, sheet_name="CaseMeansWide", index=False)
        df_case_means_long.to_excel(writer, sheet_name="CaseMeansLong", index=False)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)

    print(f"Saved Excel     : {out_excel}")


def main():
    case_voxel_tables = []
    all_mean_long = []
    all_mean_wide = []

    successful_pids = []

    for pid in CFG["pids"]:
        try:
            df_vox_wide, df_mean_long, df_mean_wide = process_one_case(pid, CFG)
            case_voxel_tables.append(df_vox_wide)
            all_mean_long.append(df_mean_long)
            all_mean_wide.append(df_mean_wide)
            successful_pids.append(pid)
        except Exception as e:
            print(f"[ERROR] PID {pid}: {e}")

    if len(case_voxel_tables) == 0:
        print("No cases were processed successfully.")
        return

    df_case_means_long = pd.concat(all_mean_long, ignore_index=True)
    df_case_means_wide = pd.concat(all_mean_wide, ignore_index=True)

    # 保存時は成功症例に合わせる
    cfg_save = dict(CFG)
    cfg_save["pids"] = successful_pids

    save_outputs(case_voxel_tables, df_case_means_long, df_case_means_wide, cfg_save)
    plot_violin_from_wide_tables(case_voxel_tables, CFG)

    print("\nDone.")
    print(df_case_means_wide.head())
    print(df_case_means_long.head())


if __name__ == "__main__":
    main()